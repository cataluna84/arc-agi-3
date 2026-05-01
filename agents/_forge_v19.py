# =====================================================================
# FORGE v19 — v18 base + 4 targeted bug fixes
#
# Fixes applied on top of v18:
#
# FIX 1: _visited_hashes was never initialized in __init__ — reward
#         signal was broken: always gave +1.5 for ANY hash change,
#         never penalizing loops. Now properly tracks and deduplicates.
#
# FIX 2: CLTI frame extraction used get_pixels() which is inconsistent
#         with _raw() (which reads frame[-1] from perform_action).
#         Now uses perform_action result frames throughout, so injected
#         expert demos have correct state representations.
#
# FIX 3: BFS hidden retry used 3 RESET calls instead of 2, landing
#         in a different initial state than the first pass scan,
#         causing the retry to search from a mismatched baseline.
#
# FIX 4: Epsilon always reset to 0.15 on level change even when BFS
#         already solved the level. Now only resets if BFS failed,
#         preserving learned exploration for CNN fallback.
# =====================================================================
import copy
import glob
import hashlib
import importlib.util
import logging
import os
import random
import time
import traceback
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)

# ==================== BFS SOLVER ====================
def _fast_deepcopy(game):
    """Deepcopy game object, skipping the camera (rendering-only, never mutates)."""
    camera = game._camera
    game._camera = None
    g = copy.deepcopy(game)
    game._camera = camera
    g._camera = camera
    return g
    
class BFSSolver:
    """Offline BFS solver using direct game class instantiation."""

    def _detect_bg(self, frame):
        """Detect true background by sampling border pixels — more reliable than argmax."""
        border = np.concatenate([
            frame[0, :], frame[-1, :], frame[:, 0], frame[:, -1]
        ])
        cnt = np.bincount(border.flatten(), minlength=16)
        return int(cnt.argmax())
        
    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}  # level_idx → action list
        self.timed_out_levels = set()
        self.outside_goal_heuristic = None  # cached if model region detected at L0

    def load(self):
        """Load the game class from source."""
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS: Failed to load game class: {e}")
            return False
   
    def _perform_and_drain(self, game, ai, max_drain=5, drain=True):
        try:
            r = game.perform_action(ai, raw=True)
        except Exception as e:
            logger.warning(f"BFS drain: initial perform_action failed: {e}")
            raise
        if not drain or not r.frame:
            return r
    
        prev_frame = np.array(r.frame[-1])
        for _ in range(max_drain):
            try:
                r2 = game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
            except:
                break
            if not r2.frame:
                break
            curr_frame = np.array(r2.frame[-1])
            if np.array_equal(curr_frame, prev_frame):
                break
            r = r2
            prev_frame = curr_frame
        return r

    def _analyse_demo(self, frames_and_actions):
        """Analyse a demonstration (sequence of frame, action pairs) to extract:
        - Which colors are player-controlled (move in response to actions)
        - Which colors are passive targets (stationary until win)
        - What the win condition looks like structurally
        
        Returns a demo_model dict with this information.
        """
        if len(frames_and_actions) < 2:
            return None
        
        bg = int(np.bincount(
            frames_and_actions[0][0].flatten(), minlength=16).argmax())
        
        # Action direction vectors
        action_dirs = {1: (0,-1), 2: (0,1), 3: (-1,0), 4: (1,0)}
        
        def get_centroids(frame):
            result = {}
            for c in range(16):
                if c == bg: continue
                mask = (frame == c)
                n = int(np.sum(mask))
                if n < 4: continue
                ys, xs = np.where(mask)
                result[c] = (float(np.mean(xs)), float(np.mean(ys)), n)
            return result
        
        # Track per-color movement correlation with action direction
        # player-controlled colors move in the action direction
        color_action_corr = {}  # color -> list of (expected_dx, actual_dx, expected_dy, actual_dy)
        color_movement = {}     # color -> total movement across all steps
        
        prev_frame, _ = frames_and_actions[0]
        prev_centroids = get_centroids(prev_frame)
        
        for frame, action in frames_and_actions[1:]:
            curr_centroids = get_centroids(frame)
            adx, ady = action_dirs.get(action, (0, 0))
            
            for c in prev_centroids:
                if c not in curr_centroids:
                    continue
                actual_dx = curr_centroids[c][0] - prev_centroids[c][0]
                actual_dy = curr_centroids[c][1] - prev_centroids[c][1]
                movement = abs(actual_dx) + abs(actual_dy)
                
                if c not in color_action_corr:
                    color_action_corr[c] = []
                    color_movement[c] = 0
                color_movement[c] += movement
                
                # Does this color move in the action direction?
                if movement > 1:
                    if adx != 0:
                        corr = np.sign(actual_dx) == np.sign(adx)
                    elif ady != 0:
                        corr = np.sign(actual_dy) == np.sign(ady)
                    else:
                        corr = False
                    color_action_corr[c].append(corr)
            
            prev_frame = frame
            prev_centroids = curr_centroids
        
        # Track pixel count stability per color
        # Player colors maintain consistent pixel counts
        # Target colors that get overlapped show sudden pixel count changes at win step
        color_pixel_counts = {}  # color -> list of pixel counts across frames
        for frame, action in frames_and_actions:
            c_counts = {}
            for c in range(16):
                if c == bg: continue
                n = int(np.sum(frame == c))
                if n >= 4:
                    c_counts[c] = n
            for c, n in c_counts.items():
                if c not in color_pixel_counts:
                    color_pixel_counts[c] = []
                color_pixel_counts[c].append(n)
    
        player_colors = set()
        passive_colors = set()
        for c, corrs in color_action_corr.items():
            total_movement = color_movement.get(c, 0)
            
            # Check pixel count stability
            counts = color_pixel_counts.get(c, [])
            if len(counts) >= 2:
                count_variance = max(counts) - min(counts)
                # High variance in pixel count = color appears/disappears = target being overlapped
                count_stable = count_variance < max(counts) * 0.3
            else:
                count_stable = True
    
            if not corrs:
                if total_movement < 1:
                    passive_colors.add(c)
                continue
            corr_rate = sum(corrs) / len(corrs)
            if corr_rate > 0.5 and total_movement > 5 and count_stable:
                player_colors.add(c)
            elif corr_rate < 0.3 or not count_stable:
                passive_colors.add(c)
        
        # Win frame analysis
        win_frame = frames_and_actions[-1][0]
        init_frame = frames_and_actions[0][0]
        win_centroids = get_centroids(win_frame)
        init_centroids = get_centroids(init_frame)
        
        # What changed at the win step vs second-to-last step?
        pre_win_frame = frames_and_actions[-2][0]
        pre_win_centroids = get_centroids(pre_win_frame)
        
        win_changes = {}  # color -> (pre_win_pos, win_pos)
        for c in pre_win_centroids:
            if c not in win_centroids:
                continue
            dx = abs(win_centroids[c][0] - pre_win_centroids[c][0])
            dy = abs(win_centroids[c][1] - pre_win_centroids[c][1])
            if dx + dy > 2:
                win_changes[c] = (
                    (pre_win_centroids[c][0], pre_win_centroids[c][1]),
                    (win_centroids[c][0], win_centroids[c][1])
                )
        
       # Win conditions: which player colors moved TOWARD passive colors at the win step?
        # Compare pre-win distance vs post-win distance for each (player, passive) pair
        win_conditions = []
        for pc in player_colors:
            if pc not in win_centroids or pc not in pre_win_centroids:
                continue
            for tc in passive_colors:
                if tc not in win_centroids or tc not in pre_win_centroids:
                    continue
                # Distance before and after win step
                pre_dist = (abs(pre_win_centroids[pc][0] - pre_win_centroids[tc][0]) +
                           abs(pre_win_centroids[pc][1] - pre_win_centroids[tc][1]))
                post_dist = (abs(win_centroids[pc][0] - win_centroids[tc][0]) +
                            abs(win_centroids[pc][1] - win_centroids[tc][1]))
                # Player color moved toward passive color at win step
                if post_dist < pre_dist and post_dist < 15:
                    win_conditions.append((pc, tc))
        
        # Pixel-level win signature: what transformation happened?
        changed_mask = init_frame != win_frame
        n_changed = int(np.sum(changed_mask))
        
        return {
            'player_colors': player_colors,
            'passive_colors': passive_colors,
            'win_conditions': win_conditions,  # (player_color, target_color) pairs
            'win_centroids': win_centroids,
            'init_centroids': init_centroids,
            'bg': bg,
            'n_changed': n_changed,
            'win_frame': win_frame,
            'init_frame': init_frame,
        }
     
    def _state_hash(self, g, frame, hidden_fields=None, transient_fields=None):
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        ignore = {'_action_count', '_full_reset', '_action_complete', '_debug', '_seed'}
        if transient_fields:
            ignore.update(transient_fields)
        extras = []
        for k, v in g.__dict__.items():
            if k.startswith('__') or k in ignore:
                continue
            if isinstance(v, (int, float, bool)):
                extras.append(f"{k}={v}")
            elif isinstance(v, (set, frozenset)) and len(v) < 50:
                extras.append(f"{k}={sorted(str(i) for i in v)}")
        if extras:
            eh = hashlib.md5("|".join(sorted(extras)).encode()).hexdigest()[:12]
            return fh + "|" + eh
        return fh

    def _probe_hidden_fields(self, game, actions):
        """Dynamic state probing — discover which scalar fields change per action.
        Returns list of field names that are hidden state (change without pixel change)."""
        if not actions:
            return []
        initial = {}
        for k, v in game.__dict__.items():
            if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                initial[k] = v

        changing_fields = set()
        frame0 = game.get_pixels(0, 0, 64, 64)
        for act_id, data in actions[:10]:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai, raw=True)
            except:
                continue
            f = g.get_pixels(0, 0, 64, 64)
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing_fields.add(k)

        hidden = []
        for f in changing_fields:
            if f.startswith('_') and f not in ('_current_level_index', '_score'):
                continue
            hidden.append(f)
        return sorted(hidden)

    def _detect_transient_fields(self, game, actions):
        """Detect scalar fields that change on every action (e.g. budget counters,
        monotonic clocks). These add no state-distinguishing value to the hash and
        cause state space explosion if included."""
        if not actions:
            return set()
        initial = {k: v for k, v in game.__dict__.items()
                   if isinstance(v, (int, float, bool)) and not k.startswith('__')
                   and k not in ('_action_count', '_full_reset', '_action_complete')}
        # Track how many sampled actions changed each field
        changed_count = {k: 0 for k in initial}
        n_sampled = 0
        for act_id, data in actions[:min(12, len(actions))]:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai, raw=True)
            except:
                continue
            n_sampled += 1
            for k in initial:
                if getattr(g, k, initial[k]) != initial[k]:
                    changed_count[k] += 1
        # Also sample click actions so click-triggered transients are detected
        if hasattr(game, '_get_valid_actions'):
            try:
                for va in game._get_valid_actions()[:4]:
                    g = copy.deepcopy(game)
                    try:
                        g.perform_action(va, raw=True)
                    except:
                        continue
                    n_sampled += 1
                    for k in initial:
                        if getattr(g, k, initial[k]) != initial[k]:
                            changed_count[k] += 1
            except:
                pass            
        if n_sampled == 0:
            return set()
        # A field is transient if it changed in every sampled action
        # Exclude monotonic counters (always decrease/increase) but keep boolean flags
        # Boolean flags encode meaningful state (e.g. which object is selected)
        transient = set()
        for k, cnt in changed_count.items():
            if cnt != n_sampled:
                continue
            v = initial[k]
            if isinstance(v, bool):
                continue  # boolean flags are meaningful state, never transient
            transient.add(k)
        if transient:
            logger.info(f"BFS: detected transient fields (excluded from hash): {transient}")
        return transient
    
    def _build_goal_heuristic(self, f_init, f_prev_win, demo_model=None):

        def count_indicators(game):
            try:
                total, satisfied = 0, 0
                for av in game.__dict__.values():
                    if not isinstance(av, dict): continue
                    for v in av.values():
                        if not isinstance(v, list): continue
                        for item in v:
                            if hasattr(item, 'is_visible') and hasattr(item, 'pixels'):
                                total += 1
                                if item.is_visible: satisfied += 1
                return total, satisfied
            except:
                return 0, 0
    
        # Measure baseline at level start so heuristic is relative, not absolute
        _baseline_unsatisfied = [None]
        _cost_per_indicator = [5]  # default: assume ~5 actions per indicator
        if self.game_cls:
            try:
                test = self.game_cls()
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                t, s = count_indicators(test)
                _baseline_unsatisfied[0] = t - s
            except:
                pass
            # Calibrate cost_per_indicator from known solutions
            # avg actions per indicator satisfied across all solved levels
            if self.solutions:
                try:
                    total_actions = 0
                    total_satisfied = 0
                    cal_game = self.game_cls()
                    cal_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    cal_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    prev_unsat = _baseline_unsatisfied[0] or 0
                    for si in sorted(self.solutions.keys()):
                        sol = self.solutions[si]
                        for act_id, data in sol:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            cal_game.perform_action(ai, raw=True)
                        t2, s2 = count_indicators(cal_game)
                        curr_unsat = t2 - s2
                        satisfied_this_level = max(0, prev_unsat - curr_unsat)
                        total_actions += len(sol)
                        total_satisfied += satisfied_this_level
                        prev_unsat = curr_unsat
                    if total_satisfied > 0:
                        _cost_per_indicator[0] = max(3, total_actions / total_satisfied)
                        logger.info(f"BFS heuristic: calibrated cost_per_indicator={_cost_per_indicator[0]:.1f} ({total_actions} actions / {total_satisfied} satisfied)")
                except Exception as e:
                    logger.warning(f"BFS heuristic: calibration failed: {e}")

        def introspection_heuristic(f, game=None):
            if game is None:
                return 0
            try:
                total, satisfied = count_indicators(game)
                if total == 0:
                    return 0
                current_unsatisfied = total - satisfied
                if _baseline_unsatisfied[0] is not None and _baseline_unsatisfied[0] == 0:
                    return 0
                return current_unsatisfied
            except:
                return 0
    
        # Validate using the actual level start frame, not L0
        # f_init is the frame at the start of this level
        if f_init is not None:
            try:
                # Measure indicator count at init state (L0 fresh game)
                test_init = self.game_cls()
                test_init.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                test_init.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                total_test, sat_init = count_indicators(test_init)
                unsatisfied_init = total_test - sat_init

                if total_test > 0 and np.sum(f_init != f_prev_win) > 0:
                    # Measure indicator count at win state by replaying the prev solution
                    # This uses a throwaway game — no real actions consumed
                    if self.solutions:
                        test_win = self.game_cls()
                        test_win.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        test_win.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        last_sol_idx = max(self.solutions.keys())
                        try:
                            for si in range(last_sol_idx + 1):
                                sol_actions = self.solutions[si]
                                # Stop BEFORE the last action of the last level
                                # so we measure state just before winning, not after
                                is_last_level = (si == last_sol_idx)
                                actions_to_run = sol_actions[:-1] if is_last_level else sol_actions
                                for act_id, data in actions_to_run:
                                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                    test_win.perform_action(ai, raw=True)
                            total_win, sat_win = count_indicators(test_win)
                            # Use total_win not total_test — indicator count grows each level
                            unsatisfied_win = total_win - sat_win
                            # Also recompute unsatisfied_init using same total basis
                            # by measuring at the level start, not L0
                            test_init2 = self.game_cls()
                            test_init2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                            test_init2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                            for si in range(last_sol_idx):
                                for act_id, data in self.solutions[si]:
                                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                    test_init2.perform_action(ai, raw=True)
                            total_init2, sat_init2 = count_indicators(test_init2)
                            unsatisfied_init = total_init2 - sat_init2
                            if unsatisfied_win < unsatisfied_init:
                                logger.info(f"BFS heuristic: introspection varies init={unsatisfied_init} win={unsatisfied_win}, using it")
                                return introspection_heuristic
                            else:
                                logger.info(f"BFS heuristic: introspection not useful init={unsatisfied_init} win={unsatisfied_win}, skipping to proximity")
                        except Exception as e:
                            logger.warning(f"BFS heuristic win-state probe failed: {e}")
                            # Can't verify — fall through to proximity
                    else:
                        # No solutions yet (L0) — use indicator count directly
                        if unsatisfied_init > 0:
                            logger.info(f"BFS heuristic: L0 introspection unsatisfied={unsatisfied_init}, using it")
                            return introspection_heuristic
            except:
                pass
    
        # Build a rare-object proximity heuristic from the win frame delta.
        # Hypothesis: objects that changed position between init and win frame
        # are the interactive ones. Minimise distance between them.
        try:
            f_init_c = np.clip(f_init, 0, 15).astype(np.int64).flatten()
            f_prev_win_c = np.clip(f_prev_win, 0, 15).astype(np.int64).flatten()
            cnt_init = np.bincount(f_init_c, minlength=16)[:16]
            cnt_win  = np.bincount(f_prev_win_c, minlength=16)[:16]
            bg = int(cnt_init.argmax())
            total_pixels = len(f_init_c)
            # Reshape back to 2D for centroid calculations
            s1 = int(np.sqrt(len(f_init_c)))
            s2 = int(np.sqrt(len(f_prev_win_c)))
            f_init_2d = f_init_c.reshape(s1, s1) if s1*s1 == len(f_init_c) else f_init_c.reshape(1,-1)
            f_prev_win_2d = f_prev_win_c.reshape(s2, s2) if s2*s2 == len(f_prev_win_c) else f_prev_win_c.reshape(1,-1)
    
            # Score each color by rarity and how much it changed between init and win
            color_scores = {}
            for c in range(16):
                if c == bg: continue
                if cnt_init[c] == 0 and cnt_win[c] == 0: continue
                rarity = 1.0 - (cnt_init[c] / total_pixels)  # rare = high score
                changed = abs(int(cnt_win[c]) - int(cnt_init[c]))
                moved = (cnt_init[c] != cnt_win[c]) if cnt_init[c] > 0 else False
                color_scores[c] = rarity * (1 + changed + (2 if moved else 0))
    
            if not color_scores:
                logger.info("BFS heuristic: no colors found, uniform cost")
                return lambda f, game=None: 0
    
            # Top 2 most interactive colors = likely player and target
            sorted_colors = sorted(color_scores, key=color_scores.get, reverse=True)
            interactive = sorted_colors[:2]
    
            # If only one interactive color, use distance to win-frame centroid
            def get_centroid(frame, color):
                frame = np.clip(frame, 0, 15).flatten()
                s = int(np.sqrt(len(frame)))
                frame = frame.reshape(s, s) if s*s == len(frame) else frame.reshape(1,-1)
                mask = (frame == color)
                if not np.any(mask): return None
                ys, xs = np.where(mask)
                return (float(np.mean(xs)), float(np.mean(ys)))
    
            if len(interactive) == 1:
                c = interactive[0]
                win_pos = get_centroid(f_prev_win, c)
                if win_pos is None:
                    return lambda f, game=None: 0
                def single_heuristic(f, game=None, _c=c, _wp=win_pos):
                    pos = get_centroid(f, _c)
                    if pos is None: return 0
                    return abs(pos[0] - _wp[0]) + abs(pos[1] - _wp[1])
                logger.info(f"BFS heuristic: single-color proximity to win pos, color={c}")
                return single_heuristic
    
            c1, c2 = interactive[0], interactive[1]
            logger.info(f"BFS heuristic: rare-object proximity, colors={c1},{c2} scores={color_scores.get(c1,0):.2f},{color_scores.get(c2,0):.2f}")
    
            def proximity_heuristic(f, game=None, _c1=c1, _c2=c2):
                pos1 = get_centroid(f, _c1)
                pos2 = get_centroid(f, _c2)
                if pos1 is None or pos2 is None: return 0
                return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
            return proximity_heuristic
    
        except Exception as e:
            logger.warning(f"BFS heuristic build failed: {e}")
    
        logger.info("BFS heuristic: no indicators found, uniform cost")
        return lambda f, game=None: 0

    def _scan_actions(self, game, f0, bg):
        """Scan for effective actions. Returns list of (action_id, data)."""
        avail = game._available_actions
        actions = []
        for a in [a for a in avail if a <= 5]:
            actions.append((a, None))
        if 6 in avail:
            seen_effects = set()
            if hasattr(game, '_get_valid_actions'):
                try:
                    valid = game._get_valid_actions()
                    for ai_obj in valid:
                        act_id = ai_obj.id._value_ if hasattr(ai_obj.id, '_value_') else int(ai_obj.id)
                        if act_id == 6:
                            g = copy.deepcopy(game)
                            try:
                                r = g.perform_action(ai_obj, raw=True)
                                if r.frame:
                                    f = np.array(r.frame[-1])
                                    diff = np.sum(f0 != f)
                                    if diff > 0:
                                        eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                        if eh not in seen_effects:
                                            seen_effects.add(eh)
                                            actions.append((6, ai_obj.data))
                            except:
                                pass
                except:
                    pass
            if not seen_effects:
                t0 = time.time()
                for y in range(0, 64, 2):
                    if time.time() - t0 > self.scan_timeout:
                        break
                    for x in range(0, 64, 2):
                        if f0[y, x] == bg:
                            continue
                        g = copy.deepcopy(game)
                        try:
                            r = g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y}), raw=True)
                            if not r.frame:
                                continue
                            f = np.array(r.frame[-1])
                            diff = np.sum(f0 != f)
                            if diff > 0:
                                effect_hash = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                if effect_hash not in seen_effects:
                                    seen_effects.add(effect_hash)
                                    actions.append((6, {'x': x, 'y': y}))
                        except:
                            pass
        return actions

    def _get_click_actions_for_state(self, game, frame, bg, cached_non_bg_positions):
        """
        Only generate click actions for non-background pixels
        that are within 2 cells of any changed pixel since last step.
        Falls back to all non-bg pixels if no prior frame available.
        """
        if hasattr(game, '_get_valid_actions'):
            try:
                valid = game._get_valid_actions()
                result = []
                seen = set()
                for ai_obj in valid:
                    act_id = ai_obj.id._value_ if hasattr(ai_obj.id, '_value_') else int(ai_obj.id)
                    if act_id == 6 and ai_obj.data:
                        key = (ai_obj.data.get('x'), ai_obj.data.get('y'))
                        if key not in seen:
                            seen.add(key)
                            result.append((6, ai_obj.data))
                return result
            except:
                pass
    
        # Fallback: only non-background pixels
        actions = []
        seen_effects = set()
        for pos in cached_non_bg_positions:
            x, y = pos
            if frame[y, x] == bg:
                continue
            actions.append((6, {'x': x, 'y': y}))
        return actions
        
    def _probe_mover_target_colors(self, game):
        g = copy.deepcopy(game)
        avail = [a for a in game._available_actions if 1 <= a <= 4]
        if not avail:
            return set(), set()
        r0 = g.perform_action(ActionInput(id=GameAction.from_id(avail[0])), raw=True)
        if not r0.frame:
            return set(), set()
        f0 = np.array(r0.frame[-1])
        bg = self._detect_bg(f0)
        total_px = f0.size
    
        # NEW: detect border/padding color (second most common border color)
        border = np.concatenate([f0[0,:], f0[-1,:], f0[:,0], f0[:,-1]])
        border_cnt = np.bincount(border.flatten(), minlength=16)
        border_cnt[bg] = 0
        padding = int(border_cnt.argmax()) if border_cnt.max() > 0 else -1
    
        # NEW: compute pixel counts to exclude large structural colors
        cnt = np.bincount(f0.flatten(), minlength=16)
        structural_threshold = total_px * 0.15  # colors >15% of frame are structural
    
        def get_centroids(frame):
            result = {}
            for c in range(16):
                if c == bg or c == padding: continue
                if cnt[c] > structural_threshold: continue  # skip structural
                mask = (frame == c)
                n = int(np.sum(mask))
                if n < 2: continue
                ys, xs = np.where(mask)
                result[c] = (float(np.mean(xs)), float(np.mean(ys)))
            return result
    
        movement = {}
        prev_c = get_centroids(f0)
        for _ in range(20):
            act = random.choice(avail)
            try:
                r2 = g.perform_action(ActionInput(id=GameAction.from_id(act)), raw=True)
            except:
                break
            if not r2.frame:
                break
            curr_c = get_centroids(np.array(r2.frame[-1]))
            for c in prev_c:
                if c in curr_c:
                    movement[c] = movement.get(c, 0.0) + abs(curr_c[c][0]-prev_c[c][0]) + abs(curr_c[c][1]-prev_c[c][1])
            prev_c = curr_c
    
        mover_colors  = {c for c, m in movement.items() if m > 5}
        target_colors = {c for c, m in movement.items() if m == 0}
    
        # Also include non-moving colors not seen during probing
        frame_cnt = np.bincount(f0.flatten(), minlength=16)
        for c in range(16):
            if c == bg or c == padding: continue
            if frame_cnt[c] == 0: continue
            if frame_cnt[c] > structural_threshold: continue  # skip structural
            if c not in movement:
                target_colors.add(c)
    
        if len(mover_colors) > 3:
            sorted_movers = sorted(mover_colors, key=lambda c: movement.get(c,0), reverse=True)
            mover_colors = set(sorted_movers[:2])
    
        return mover_colors, target_colors

    def _build_outside_goal_heuristic(self, frame):
        """
        Find two largest non-background connected regions via flood fill.
        Larger = playing field, smaller = model solution region.
        Colors appearing in both = moveable objects with known goal positions.
        No color or layout assumptions — works purely from frame topology.
        Returns a heuristic function or None if no model region detected.
        """
        try:
            border = np.concatenate([frame[0,:], frame[-1,:], frame[:,0], frame[:,-1]])
            bg = int(np.bincount(border.flatten(), minlength=16).argmax())
            non_bg = (frame != bg)
            visited = np.zeros(frame.shape, dtype=bool)
            components = []

            for sy in range(frame.shape[0]):
                for sx in range(frame.shape[1]):
                    if not non_bg[sy, sx] or visited[sy, sx]:
                        continue
                    # Iterative flood fill
                    mask = np.zeros(frame.shape, dtype=bool)
                    stack = [(sy, sx)]
                    while stack:
                        y, x = stack.pop()
                        if y < 0 or y >= frame.shape[0] or x < 0 or x >= frame.shape[1]:
                            continue
                        if visited[y, x] or not non_bg[y, x]:
                            continue
                        visited[y, x] = True
                        mask[y, x] = True
                        stack.extend([(y+1,x),(y-1,x),(y,x+1),(y,x-1)])
                    size = int(np.sum(mask))
                    if size > 10:
                        components.append((size, mask))

            if len(components) < 2:
                return None

            components.sort(key=lambda c: c[0], reverse=True)
            play_mask = components[0][1]
            model_mask = components[1][1]

            # Identify playing field color — most common non-bg color in play region
            play_cnt = np.bincount(frame[play_mask].flatten(), minlength=16)
            play_cnt[bg] = 0
            field_color = int(play_cnt.argmax())

            # Identify model field color — most common non-bg color in model region
            model_cnt = np.bincount(frame[model_mask].flatten(), minlength=16)
            model_cnt[bg] = 0
            model_field_color = int(model_cnt.argmax())

            # These structural colors should be ignored everywhere
            structural = {bg, field_color, model_field_color}

            goal_map = {}
            for c in range(16):
                if c in structural:
                    continue
                model_pixels = model_mask & (frame == c)
                play_pixels  = play_mask  & (frame == c)
                if not np.any(model_pixels) or not np.any(play_pixels):
                    continue
                model_count = int(np.sum(model_pixels))
                if model_count < 4:
                    continue
                play_count = int(np.sum(play_pixels))
                # Skip colors where play region has much more pixels than model
                # — these are player/structural elements that appear larger in play
                # Goal blocks should have similar pixel counts in both regions
                if play_count > model_count * 4:
                    continue
                out_ys, out_xs = np.where(model_pixels)
                in_ys, in_xs = np.where(play_pixels)
                goal_x = float(np.mean(out_xs))
                goal_y = float(np.mean(out_ys))
                curr_x = float(np.mean(in_xs))
                curr_y = float(np.mean(in_ys))
                dist = abs(curr_x - goal_x) + abs(curr_y - goal_y)
                # Skip elongated shapes in model — rods/structural, not blocks
                # Blocks are roughly square; rods have high aspect ratio
                height = int(out_ys.max() - out_ys.min()) + 1
                width = int(out_xs.max() - out_xs.min()) + 1
                aspect = max(height, width) / max(min(height, width), 1)
                if aspect > 2.5:
                    continue

                goal_map[c] = {
                    'goal_x': goal_x,
                    'goal_y': goal_y,
                    'weight': float(np.sum(model_pixels)),
                }

            if not goal_map:
                return None

            # Verify heuristic is non-zero at current state
            total_dist = 0.0
            for c, info in goal_map.items():
                pp = play_mask & (frame == c)
                if not np.any(pp): continue
                in_ys, in_xs = np.where(pp)
                total_dist += (abs(float(np.mean(in_xs)) - info['goal_x']) +
                               abs(float(np.mean(in_ys)) - info['goal_y']))
            if total_dist < 1.0:
                return None

            logger.info(f"BFS outside-goal: {len(goal_map)} goal colors={list(goal_map.keys())} total_dist={total_dist:.1f}")

            # Normalise weights so they sum to len(goal_map)
            total_weight = sum(info['weight'] for info in goal_map.values())
            for info in goal_map.values():
                info['weight'] = (info['weight'] / total_weight) * len(goal_map) if total_weight > 0 else 1.0

            def outside_goal_heuristic(f, game=None, _gm=goal_map, _pm=play_mask):
                total = 0.0
                for c, info in _gm.items():
                    pp = _pm & (f == c)
                    if not np.any(pp):
                        total += 64.0 * info['weight']
                        continue
                    in_ys, in_xs = np.where(pp)
                    dist = (abs(float(np.mean(in_xs)) - info['goal_x']) +
                            abs(float(np.mean(in_ys)) - info['goal_y']))
                    total += dist * info['weight']
                return total

            return outside_goal_heuristic

        except Exception as e:
            logger.warning(f"BFS outside-goal heuristic error: {e}")
            return None
    
    def solve_level(self, level_idx, max_states=500000, prev_solution=None, goal_heuristic=None):
        """Find optimal solution for a level via BFS (Memory Optimised via Action Replay)."""
        if not self.game_cls:
            return None

        game = self.game_cls()
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        
        # Advance to target level by replaying previous solutions
        last_r = r0
        for prev_idx in range(level_idx):
            prev_sol = self.solutions.get(prev_idx)
            if not prev_sol:
                return None
            for act_id, data in prev_sol:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                last_r = game.perform_action(ai, raw=True)

        if not last_r.frame:
            return None
        f0 = np.array(last_r.frame[-1])
        bg = self._detect_bg(f0)
        # After computing f0 and bg:
        non_bg_positions = [(x, y) for y in range(0, 64, 2) 
                            for x in range(0, 64, 2) 
                            if f0[y, x] != bg]
        # Try solution transfer from previous level first
        if prev_solution and level_idx > 0:
            transfer_result = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer_result:
                return transfer_result

        # Phase 1: Scan for effective actions
        actions = self._scan_actions(game, f0, bg)

        # Warm-up unlock for locked initial states (sc25-type)
        if not actions:
            avail = game._available_actions
            # Try all non-reset actions as warmup, including clicks
            warmup_candidates = [a for a in avail if 1 <= a <= 5]
            # Also try click actions from _get_valid_actions if available
            if 6 in avail and hasattr(game, '_get_valid_actions'):
                try:
                    for va in game._get_valid_actions():
                        act_id = va.id._value_ if hasattr(va.id, '_value_') else int(va.id)
                        if act_id == 6:
                            g_warmup = _fast_deepcopy(game)
                            try:
                                g_warmup.perform_action(va, raw=True)
                                f_after = np.array(g_warmup.perform_action(
                                    ActionInput(id=GameAction.ACTION1), raw=True).frame[-1])
                                warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                                if warmup_actions:
                                    logger.info(f"BFS L{level_idx}: UNLOCKED with click! {len(warmup_actions)} actions")
                                    game = g_warmup; f0 = f_after; actions = warmup_actions
                                    break
                            except:
                                pass
                except:
                    pass
            if not actions:
                for warmup_id in [a for a in avail if a <= 4]:
                    g_warmup = _fast_deepcopy(game)
                    try:
                        g_warmup.perform_action(ActionInput(id=GameAction.from_id(warmup_id)), raw=True)
                        f_after = np.array(g_warmup.get_pixels(0, 0, 64, 64))
                        warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                        if warmup_actions:
                            logger.info(f"BFS L{level_idx}: UNLOCKED with ACTION{warmup_id}! {len(warmup_actions)} actions")
                            game = g_warmup; f0 = f_after; actions = warmup_actions
                            break
                    except:
                        pass

        logger.info(f"BFS L{level_idx}: {len(actions)} effective actions")
        if not actions:
            return None

       # ==========================================
        # Phase 2: A* with goal heuristic from prev level
        # ==========================================
        import heapq
        hidden_fields = None
        transient_fields = self._detect_transient_fields(game, actions)
        visited = set()
        h0 = self._state_hash(game, f0, None, transient_fields=transient_fields)
        visited.add(h0)
        # Pre-measure indicator baseline at this level for heuristic calibration
        def count_indicators_direct(g):
            try:
                total, satisfied = 0, 0
                for av in g.__dict__.values():
                    if not isinstance(av, dict): continue
                    for v in av.values():
                        if not isinstance(v, list): continue
                        for item in v:
                            if hasattr(item, 'is_visible') and hasattr(item, 'pixels'):
                                total += 1
                                if item.is_visible: satisfied += 1
                return total, satisfied
            except:
                return 0, 0

        _level_total, _level_satisfied = count_indicators_direct(game)
        _level_unsatisfied = _level_total - _level_satisfied
        logger.info(f"BFS L{level_idx}: indicator baseline total={_level_total} satisfied={_level_satisfied} unsatisfied={_level_unsatisfied}")

        # If all indicators already satisfied at level start, heuristic will be flat
        # Override with None to force distance heuristic
        if _level_unsatisfied == 0 and goal_heuristic is not None:
            logger.info(f"BFS L{level_idx}: all indicators satisfied at level start, disabling introspection heuristic")
            goal_heuristic = None
        

        hfn = goal_heuristic if goal_heuristic is not None else (lambda f, game=None: 0)
        _hfn_uses_game = goal_heuristic is not None
        counter = 0
        effective_timeout = min(self.bfs_timeout + level_idx * 30, 300)
        bfs_timeout = min(10, effective_timeout * 0.15)  # short BFS phase first

        # Phase 2a: plain BFS first — fast for small state spaces, no heuristic bias
        bfs_queue = deque()
        bfs_visited = set()
        bfs_queue.append(([], _fast_deepcopy(game)))
        bfs_visited.add(h0)
        t0 = time.time()
        explored = 0

        while bfs_queue and (time.time() - t0) < bfs_timeout:
            hist_b, node_b = bfs_queue.popleft()
            for act_id, data in actions:
                g2 = _fast_deepcopy(node_b)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
                except:
                    continue
                explored += 1
                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f, hidden_fields, transient_fields=transient_fields)
                if h in bfs_visited:
                    continue
                bfs_visited.add(h)
                new_hist = hist_b + [(act_id, data)]
                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    elapsed = time.time() - t0
                    logger.info(f"BFS L{level_idx}: SOLVED (BFS) in {len(new_hist)} actions ({explored} explored, {elapsed:.1f}s)")
                    self.solutions[level_idx] = new_hist
                    return new_hist
                bfs_queue.append((new_hist, _fast_deepcopy(g2)))

        bfs_elapsed = time.time() - t0
        logger.info(f"BFS L{level_idx}: BFS phase done ({explored} explored in {bfs_elapsed:.1f}s), switching to A*")

        # Phase 2b: A* with heuristic for remaining time
        visited = bfs_visited  # reuse visited states from BFS phase
        pq = []
        # Seed A* from all frontier states in BFS queue
        for hist_b, node_b in bfs_queue:
            if not hist_b:
                continue
            try:
                # Get current frame by replaying last action on a copy
                last_act_id, last_data = hist_b[-1]
                g_seed = _fast_deepcopy(node_b)
                h_val = hfn(f0, g_seed if _hfn_uses_game else None) * 10
            except:
                h_val = 0
            counter += 1
            heapq.heappush(pq, (len(hist_b) + h_val, len(hist_b), counter, hist_b, node_b))
        
        # If BFS queue was empty or seeding failed, start fresh from initial state
        if not pq:
            pq = [(hfn(f0, game) * 10, 0, counter, [], _fast_deepcopy(game))]

        
        # Progress tracking for adaptive timeout
        last_progress_check = time.time()
        states_at_last_check = 0

        while pq and explored < max_states and (time.time() - t0) < effective_timeout:
            f_score, g_score, _, hist, node_state = heapq.heappop(pq)

            # Progress check every 5 seconds
            now = time.time()
            if now - last_progress_check > 5.0:
                states_since_check = explored - states_at_last_check
                queue_size = len(pq)
                if queue_size < 100 and states_since_check > 1000:
                    self.bfs_timeout = min(self.bfs_timeout + 30, 300)
                    logger.info(f"BFS L{level_idx}: queue near exhaustion, extending timeout to {self.bfs_timeout}s")
                elif queue_size > 50000 and states_since_check < 500:
                    logger.info(f"BFS L{level_idx}: large state space, cutting to CNN early")
                    break
                last_progress_check = now
                states_at_last_check = explored

            for act_id, data in actions:
                g2 = _fast_deepcopy(node_state)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
                except:
                    continue
                explored += 1

                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f, hidden_fields, transient_fields=transient_fields)
                if h in visited:
                    continue
                visited.add(h)

                new_hist = hist + [(act_id, data)]
                new_g = g_score + 1

                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    elapsed = time.time() - t0
                    logger.info(f"BFS L{level_idx}: SOLVED (A*) in {len(new_hist)} actions ({explored} explored, {elapsed:.1f}s)")
                    self.solutions[level_idx] = new_hist
                    return new_hist

                h_val = hfn(f, g2 if _hfn_uses_game else None) * 10
                counter += 1
                # Store state dict on heap, not the game object
                heapq.heappush(pq, (new_g + h_val, new_g, counter, new_hist, _fast_deepcopy(g2)))

        elapsed_first = time.time() - t0
        logger.info(f"BFS L{level_idx}: first pass timeout ({explored} explored, {len(visited)} unique, {elapsed_first:.1f}s)")
        self.timed_out_levels.add(level_idx)
        # Dynamic action rescan BFS — triggers when state space exhausted quickly
        # indicating actions expand as state evolves (e.g. flood fill games)
        exhausted_quickly = len(pq) == 0 and elapsed_first < self.bfs_timeout * 0.5
        if exhausted_quickly:
            logger.info(f"BFS L{level_idx}: queue exhausted early — retrying with dynamic action rescan")
            visited_d = set()
            _rescan_game = _fast_deepcopy(game)
            visited_d.add(self._state_hash(_rescan_game, f0, hidden_fields, transient_fields=transient_fields))
            queue_d = deque()
            queue_d.append(([], 0, _rescan_game))
            t0_d = time.time()
            explored_d = 0
            remaining_d = max(30, self.bfs_timeout - elapsed_first)
            current_actions = list(actions)

            while queue_d and explored_d < max_states * 10 and (time.time() - t0_d) < remaining_d:
                hist_d, depth_d, node_game_d = queue_d.popleft()

                for act_id, data in current_actions:
                    g2_d = _fast_deepcopy(node_game_d)
                    try:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        r = g2_d.perform_action(ai, raw=True)
                    except:
                        continue
                    explored_d += 1
                    if not r.frame:
                        continue
                    f2_d = np.array(r.frame[-1])
                    h_d = self._state_hash(g2_d, f2_d, hidden_fields, transient_fields=transient_fields)
                    if h_d in visited_d:
                        continue
                    visited_d.add(h_d)
                    # Rescan from child state to find newly unlocked actions
                    try:
                        new_acts = self._scan_actions(g2_d, f0, bg)
                        added = [a for a in new_acts if a not in current_actions]
                        if added:
                            logger.info(f"BFS L{level_idx}: rescan found {len(added)} new actions at depth {depth_d}")
                            current_actions.extend(added)
                    except:
                        pass
                    new_hist_d = hist_d + [(act_id, data)]
                    if r.levels_completed > level_idx or g2_d._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: SOLVED (dynamic rescan) in {len(new_hist_d)} actions ({explored_d} explored)")
                        self.solutions[level_idx] = new_hist_d
                        return new_hist_d
                    if depth_d < 30:
                        queue_d.append((new_hist_d, depth_d + 1, g2_d))

            logger.info(f"BFS L{level_idx}: dynamic rescan also failed ({explored_d} explored)")

        # Smart early exit — game may be too expensive to BFS
        if explored < 20 and elapsed_first > 10.0:
            logger.info(f"BFS L{level_idx}: early exit (only {explored} explored in {elapsed_first:.1f}s) — handing off to CNN")
            return None

        # If too few unique states found → hidden state detected → retry with probed fields
        if explored > 0 and (len(visited) < 200 or explored / len(visited) > 5) and elapsed_first < self.bfs_timeout * 0.8:
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                logger.info(f"BFS L{level_idx}: RETRY with hidden fields: {hidden_fields}")

                # FIX 3: Use exactly 2 RESET calls (not 3) to match the first pass baseline
                game2 = self.game_cls()
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r2 = game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)

                for prev_idx in range(level_idx):
                    prev_sol = self.solutions.get(prev_idx)
                    if not prev_sol:
                        return None
                    for act_id, data in prev_sol:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        last_r2 = game2.perform_action(ai, raw=True)

                if not last_r2.frame:
                    return None
                f0_2 = np.array(last_r2.frame[-1])
                h0_2 = self._state_hash(game2, f0_2, hidden_fields, transient_fields=transient_fields)

                base_game2 = _fast_deepcopy(game2)
                visited2 = set()
                visited2.add(h0_2)
                queue2 = deque()
                queue2.append(([], 0, base_game2))

                t0_2 = time.time()
                explored2 = 0
                remaining = max(30, self.bfs_timeout - elapsed_first)

                while queue2 and explored2 < max_states and (time.time() - t0_2) < remaining:
                    hist, depth, node_game2 = queue2.popleft()

                    for act_id, data in actions:
                        g2 = _fast_deepcopy(node_game2)
                        try:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        explored2 += 1

                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, hidden_fields, transient_fields=transient_fields)
                        if h in visited2:
                            continue
                        visited2.add(h)

                        new_hist = hist + [(act_id, data)]

                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            logger.info(f"BFS L{level_idx}: SOLVED (hidden retry) in {len(new_hist)} actions ({explored2} explored)")
                            self.solutions[level_idx] = new_hist
                            return new_hist

                        if depth < 30:
                            queue2.append((new_hist, depth + 1, g2))

                logger.info(f"BFS L{level_idx}: hidden retry also failed ({explored2} explored, {len(visited2)} unique)")

        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        """Transfer previous level's solution to current level."""
        try:
            # Try executing prev solution directly
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (direct replay, {i+1} actions)")
                        sol = prev_solution[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

            # Try object-relative transfer
            prev_game = self.game_cls()
            prev_game.set_level(level_idx - 1)
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r_prev.frame:
                return None
            f0 = np.array(r_prev.frame[-1])
            bg = self._detect_bg(f0)

            def get_objects(frame, bg_c):
                objs = []
                for c in range(16):
                    if c == bg_c:
                        continue
                    mask = (frame == c)
                    npix = int(np.sum(mask))
                    if npix < 2:
                        continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)), 'cy': float(np.mean(ys)), 'n': npix})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))

            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)

            if not objs_prev or not objs_curr:
                return None

            matched = []
            for op in objs_prev:
                best = None
                best_dist = float('inf')
                for oc in objs_curr:
                    if oc['color'] == op['color'] and abs(oc['n'] - op['n']) < max(op['n'], oc['n']) * 0.5:
                        d = abs(oc['cx'] - op['cx']) + abs(oc['cy'] - op['cy'])
                        if d < best_dist:
                            best_dist = d
                            best = oc
                if best:
                    matched.append((op, best))

            if not matched:
                return None

            dx = np.mean([m[1]['cx'] - m[0]['cx'] for m in matched])
            dy = np.mean([m[1]['cy'] - m[0]['cy'] for m in matched])

            transferred = []
            for act_id, data in prev_solution:
                if data and 'x' in data:
                    new_data = dict(data)
                    new_data['x'] = max(0, min(63, int(data['x'] + dx)))
                    new_data['y'] = max(0, min(63, int(data['y'] + dy)))
                    transferred.append((act_id, new_data))
                else:
                    transferred.append((act_id, data))

            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(transferred):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (offset dx={dx:.0f},dy={dy:.0f}, {i+1} actions)")
                        sol = transferred[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

        except Exception as e:
            logger.warning(f"BFS transfer failed: {e}")
        return None


def find_game_source_and_class(game_id, arc_env=None):
    """Find the game .py file and class name."""
    import re

    # game_id format: sk48-d8078629
    # file lives at: .../environment_files/sk48/d8078629/sk48.py
    parts = game_id.split('-', 1)
    gid = parts[0]                          # e.g. sk48
    guid_suffix = parts[1] if len(parts) > 1 else ''  # e.g. d8078629

    # Primary: competition path on Kaggle
    competition_path = (
        f"/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
        f"/environment_files/{gid}/{guid_suffix}/{gid}.py"
    )
    if os.path.exists(competition_path):
        src = competition_path
        content = open(src).read()[:2000]
        m = re.search(r'class\s+(\w+)\s*\(', content)
        cls_name = m.group(1) if m else gid[0].upper() + gid[1:]
        logger.info(f"BFS: found game source at {src}, class={cls_name}")
        return src, cls_name

    # Fallback: broad glob search
    for pattern in [
        f"/kaggle/input/**/{gid}.py",
        f"/tmp/**/{gid}.py",
        f"/kaggle/working/**/{gid}.py",
    ]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            src = matches[0]
            content = open(src).read()[:2000]
            m = re.search(r'class\s+(\w+)\s*\(', content)
            cls_name = m.group(1) if m else gid[0].upper() + gid[1:]
            logger.info(f"BFS: found game source at {src}, class={cls_name}")
            return src, cls_name

    logger.warning(f"BFS: game source not found for {game_id}")
    return None, gid[0].upper() + gid[1:]


# ==================== CNN FALLBACK ====================

class CBAM(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__()
        s.fc1=nn.Linear(ch,max(ch//r,4)); s.fc2=nn.Linear(max(ch//r,4),ch)
        s.sp=nn.Conv2d(2,1,7,padding=3)
    def forward(s, x):
        B,C,H,W=x.shape
        w=torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2,3]))))); x=x*w.view(B,C,1,1)
        a=torch.sigmoid(s.sp(torch.cat([x.max(1,keepdim=True)[0],x.mean(1,keepdim=True)],1)))
        return x*a

class ActionEffectAttention(nn.Module):
    def __init__(s, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        s.mem_dim=mem_dim
        s.diff_enc=nn.Sequential(nn.Conv2d(1,8,8,stride=8),nn.ReLU(),nn.Conv2d(8,16,4,stride=4),nn.ReLU(),nn.Flatten(),nn.Linear(16*2*2,mem_dim))
        s.q_proj=nn.Linear(feat_dim,mem_dim)
        s.v_proj=nn.Linear(mem_dim+1+n_actions,n_actions)
        s.scale=mem_dim**0.5
    def forward(s, cnn_feat, mem_diffs, mem_actions, mem_rewards):
        B,M=mem_actions.shape
        if M==0:return torch.zeros(B,5,device=cnn_feat.device)
        keys=s.diff_enc(mem_diffs.reshape(B*M,1,64,64)).reshape(B,M,s.mem_dim)
        q=s.q_proj(cnn_feat).unsqueeze(1)
        attn=F.softmax(torch.bmm(q,keys.transpose(1,2))/s.scale,dim=-1)
        act_oh=F.one_hot(mem_actions.clamp(0,4),5).float()
        vals=torch.cat([keys,mem_rewards.unsqueeze(-1),act_oh],dim=-1)
        ctx=torch.bmm(attn,vals).squeeze(1)
        return s.v_proj(ctx)

class ForgeNet(nn.Module):
    def __init__(s, in_ch=26, g=64):
        super().__init__()
        s.g=g
        s.c1=nn.Conv2d(in_ch,32,3,padding=1);s.c2=nn.Conv2d(32,64,3,padding=1)
        s.c3=nn.Conv2d(64,128,3,padding=1);s.c4=nn.Conv2d(128,256,3,padding=1)
        s.attn=CBAM(256);s.ar=nn.Conv2d(256,64,1);s.ap=nn.MaxPool2d(4,4)
        s.af=nn.Linear(64*16*16,256);s.ah=nn.Linear(256,5);s.dr=nn.Dropout(0.15)
        s.cc1=nn.Conv2d(256,128,3,padding=1);s.cc2=nn.Conv2d(128,64,3,padding=1)
        s.cc3=nn.Conv2d(64,32,1);s.cc4=nn.Conv2d(32,1,1)
        s.gp=nn.AdaptiveAvgPool2d(1);s.gf=nn.Linear(256,64)
        s.aea=ActionEffectAttention(feat_dim=64,mem_dim=32,n_actions=5)
    def forward(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None):
        x=F.relu(s.c1(x));x=F.relu(s.c2(x));x=F.relu(s.c3(x));f=F.relu(s.c4(x))
        f=s.attn(f);af=F.relu(s.ar(f));af=s.ap(af).reshape(f.size(0),-1)
        al=s.ah(s.dr(F.relu(s.af(af))))
        cf=F.relu(s.cc1(f));cf=F.relu(s.cc2(cf));cf=F.relu(s.cc3(cf))
        cl=s.cc4(cf).reshape(f.size(0),-1)
        if mem_diffs is not None and mem_actions is not None:
            gf=s.gf(s.gp(f).reshape(f.size(0),-1))
            al=al+s.aea(gf,mem_diffs,mem_actions,mem_rewards)
        return torch.cat([al,cl],1)


def fast_objects(frame, bg):
    objs=[]
    for c in range(16):
        if c==bg:continue
        mask=(frame==c);npix=int(np.sum(mask))
        if npix<4 or npix>3000:continue
        ys,xs=np.where(mask)
        objs.append((c,float(np.mean(xs)),float(np.mean(ys)),npix))
    return objs


# ==================== AGENT ====================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(time.time()*1e6) + hash(s.game_id) % 1000000
        random.seed(seed); np.random.seed(seed%(2**32-1)); torch.manual_seed(seed%(2**32-1))
        s.start_time = time.time()
        s.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        s.G=64; s.IN=26
        s.net=None; s.opt=None
        s.buf=deque(maxlen=50000); s.buf_h=set()
        s.bsz=64; s.tfreq=10
        s.pt=None; s.pai=None; s.pr=None; s.ph=None
        s.cl=-1; s.fhist=deque(maxlen=6); s.la=0
        s.al=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5]
        s._wd=False; s._bg=0; s._wm=None
        s._aem_diffs=deque(maxlen=256); s._aem_actions=deque(maxlen=256); s._aem_rewards=deque(maxlen=256)
        s._ckpt_hash=None; s._unproductive=0; s._undo_avail=False
        s._eps=0.15; s._eps_min=0.03; s._eps_decay=0.9997
        s._prev_objs=None; s._obj_moved=0
        # FIX 1: Initialize _visited_hashes so _reward() deduplication works correctly
        s._visited_hashes = set()
        # BFS solver
        s._bfs = None
        s._bfs_solution = None
        s._bfs_step = 0
        s._bfs_tried = False

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES: s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f): return getattr(f, 'score', None) or f.levels_completed
    def _raw(s, fd): return np.array(fd.frame, dtype=np.int64)[-1]

    def _init_bfs(s):
        """Initialize BFS solver on first call."""
        src, cls = find_game_source_and_class(s.game_id, s.arc_env)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=60)
            if s._bfs.load():
                logger.info(f"BFS: loaded {cls} from {src}")
            else:
                s._bfs = None
                logger.warning(f"BFS: failed to load game class")
        else:
            logger.warning(f"BFS: game source not found for {s.game_id}")

    def _try_bfs_solve(s, level_idx):
        """Try to solve current level. For L1+, uses A* with a goal
        heuristic derived from the previous level's win frame."""
        if s._bfs is None:
            return None

        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None
        goal_heuristic = None

        # For L0, build indicator heuristic directly — no prior win frame available
        if level_idx == 0 and s._bfs.game_cls:
            try:
                def _count_indicators(g):
                    total, satisfied = 0, 0
                    for av in g.__dict__.values():
                        if not isinstance(av, dict): continue
                        for v in av.values():
                            if not isinstance(v, list): continue
                            for item in v:
                                if hasattr(item, 'is_visible') and hasattr(item, 'pixels'):
                                    total += 1
                                    if item.is_visible: satisfied += 1
                    return total, satisfied

                g_l0 = s._bfs.game_cls()
                g_l0.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                g_l0.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                total_l0, sat_l0 = _count_indicators(g_l0)
                if total_l0 > 0 and (total_l0 - sat_l0) > 0:
                    def goal_heuristic(f, game=None, _ci=_count_indicators):
                        if game is None: return 0
                        try:
                            t, s = _ci(game)
                            return t - s
                        except: return 0
                    logger.info(f"BFS L0: indicator heuristic unsatisfied={total_l0 - sat_l0}")
                else:
                    logger.info(f"BFS L0: no useful indicators (total={total_l0} sat={sat_l0}), uniform cost")
            except Exception as e:
               logger.warning(f"BFS L0: heuristic build failed: {e}")

        # Use cached outside-goal heuristic for L1+ if available — takes priority
        if s._bfs.outside_goal_heuristic is not None and goal_heuristic is None and level_idx > 0:
            goal_heuristic = s._bfs.outside_goal_heuristic
            logger.info(f"BFS L{level_idx}: using cached outside-goal heuristic")

        # For L1+, fall back to distance heuristic if no better heuristic available
        if level_idx > 0 and goal_heuristic is None:
            try:
                g_dist = s._bfs.game_cls()
                g_dist.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                g_dist.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                for pi in range(level_idx):
                    ps = s._bfs.solutions.get(pi)
                    if not ps: break
                    for act_id, data in ps:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        g_dist.perform_action(ai, raw=True)
                mover_colors, target_colors = s._bfs._probe_mover_target_colors(g_dist)
                if mover_colors and target_colors:
                    def goal_heuristic(f, game=None, _m=mover_colors, _t=target_colors):
                        centroids = {}
                        for c in range(16):
                            mask = (f == c)
                            n = int(np.sum(mask))
                            if n < 2: continue
                            ys, xs = np.where(mask)
                            centroids[c] = (float(np.mean(xs)), float(np.mean(ys)))
                        targets = [(centroids[tc][0], centroids[tc][1]) for tc in _t if tc in centroids]
                        if not targets: return 0
                        total = 0
                        for mc in _m:
                            if mc not in centroids: continue
                            mx, my = centroids[mc]
                            total += min(abs(mx - tx) + abs(my - ty) for tx, ty in targets)
                        return total
                    logger.info(f"BFS L{level_idx}: distance heuristic movers={mover_colors} targets={target_colors}")
            except Exception as e:
                logger.warning(f"BFS L{level_idx}: distance heuristic build failed: {e}")

        # Outside-goal heuristic: detect model solution region at L0 only.
        # If found, cache and reuse for all subsequent levels.
        # If not found at L0, never try again.
        if level_idx == 0:
            try:
                g_og = s._bfs.game_cls()
                g_og.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r_og = g_og.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                if last_r_og.frame:
                    f_og = np.array(last_r_og.frame[-1])
                    og_h = s._bfs._build_outside_goal_heuristic(f_og)
                    if og_h is not None:
                        h_test = og_h(f_og)
                        if h_test > 2.0:
                            s._bfs.outside_goal_heuristic = og_h
                            logger.info(f"BFS L0: outside-goal heuristic cached (h={h_test:.1f})")
                        else:
                            logger.info(f"BFS L0: outside-goal heuristic flat (h={h_test:.1f}), not caching")
                    else:
                        logger.info(f"BFS L0: no model region detected, outside-goal disabled")
            except Exception as e:
                logger.warning(f"BFS L0: outside-goal setup failed: {e}")

    
        # Before first solve attempt, check if indicators are flat at this level's
        # starting state. If so, switch to distance heuristic immediately rather
        # than wasting the entire timeout on blind search.
        if goal_heuristic is not None:
            try:
                g_check = s._bfs.game_cls()
                g_check.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                g_check.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                for pi in range(level_idx):
                    ps = s._bfs.solutions.get(pi)
                    if not ps: break
                    for act_id, data in ps:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        g_check.perform_action(ai, raw=True)
                # Count indicators at the actual level start state
                total_c, sat_c = 0, 0
                for av in g_check.__dict__.values():
                    if not isinstance(av, dict): continue
                    for v in av.values():
                        if not isinstance(v, list): continue
                        for item in v:
                            if hasattr(item, 'is_visible') and hasattr(item, 'pixels'):
                                total_c += 1
                                if item.is_visible: sat_c += 1
                unsatisfied_at_start = total_c - sat_c
                logger.info(f"BFS L{level_idx}: pre-solve indicator check: total={total_c} satisfied={sat_c} unsatisfied={unsatisfied_at_start}")
                if unsatisfied_at_start == 0 and total_c > 0:
                    logger.info(f"BFS L{level_idx}: all indicators already satisfied — switching to distance heuristic before first solve")
                    mover_colors, target_colors = s._bfs._probe_mover_target_colors(g_check)
                    if mover_colors and target_colors and len(mover_colors) <= 3:
                        def goal_heuristic(f, game=None, _m=mover_colors, _t=target_colors):
                            centroids = {}
                            sizes = {}
                            for c in range(16):
                                mask = (f == c)
                                n = int(np.sum(mask))
                                if n < 2: continue
                                ys, xs = np.where(mask)
                                centroids[c] = (float(np.mean(xs)), float(np.mean(ys)))
                                sizes[c] = n
                            # Separate movers into player (smallest) and blocks (larger)
                            present_movers = [c for c in _m if c in centroids]
                            present_targets = [(centroids[tc][0], centroids[tc][1]) for tc in _t if tc in centroids]
                            if not present_movers or not present_targets: return 0
                            # Player is the smallest mover
                            player = min(present_movers, key=lambda c: sizes.get(c, 999))
                            px, py = centroids[player]
                            # Heuristic: distance from player to nearest target
                            return min(abs(px - tx) + abs(py - ty) for tx, ty in present_targets)
                        logger.info(f"BFS L{level_idx}: using distance heuristic from start movers={mover_colors} targets={target_colors}")
                    else:
                        # Fall back to rare-object proximity from frame analysis
                        goal_heuristic = None  # solve_level will use uniform cost but at least won't waste time on flat indicator heuristic
            except Exception as e:
                logger.warning(f"BFS L{level_idx}: pre-solve indicator check failed: {e}")

        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol, goal_heuristic=goal_heuristic)
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            return sol
        
        # First attempt failed — check if heuristic was flat and retry with distance heuristic
        if level_idx in s._bfs.timed_out_levels:
            try:
                g_val = s._bfs.game_cls()
                g_val.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r_val = g_val.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                for pi in range(level_idx):
                    ps = s._bfs.solutions.get(pi)
                    if not ps: break
                    for act_id, data in ps:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        last_r_val = g_val.perform_action(ai, raw=True)
                if last_r_val.frame:
                    f_val = np.array(last_r_val.frame[-1])
                    h_vals = set()
                    h_val_hfn = goal_heuristic if goal_heuristic is not None else (lambda f, game=None: 0)
                    h_vals.add(round(h_val_hfn(f_val, g_val), 4))
                    for act_id in [a for a in g_val._available_actions if 1 <= a <= 4][:4]:
                        g2_val = copy.deepcopy(g_val)
                        r2_val = g2_val.perform_action(ActionInput(id=GameAction.from_id(act_id)), raw=True)
                        if r2_val.frame:
                            h_vals.add(round(h_val_hfn(np.array(r2_val.frame[-1]), g2_val), 4))
                    if len(h_vals) == 1:
                        logger.info(f"BFS L{level_idx}: heuristic was flat — retrying with distance heuristic")
                        mover_colors, target_colors = s._bfs._probe_mover_target_colors(g_val)
                        if mover_colors and target_colors and len(mover_colors) <= 3:
                            def dist_heuristic(f, game=None, _m=mover_colors, _t=target_colors):
                                centroids = {}
                                for c in range(16):
                                    mask = (f == c)
                                    n = int(np.sum(mask))
                                    if n < 2: continue
                                    ys, xs = np.where(mask)
                                    centroids[c] = (float(np.mean(xs)), float(np.mean(ys)))
                                targets = [(centroids[tc][0], centroids[tc][1]) for tc in _t if tc in centroids]
                                if not targets: return 0
                                total = 0
                                for mc in _m:
                                    if mc not in centroids: continue
                                    mx, my = centroids[mc]
                                    total += min(abs(mx - tx) + abs(my - ty) for tx, ty in targets)
                                return total
                            logger.info(f"BFS L{level_idx}: distance heuristic movers={mover_colors} targets={target_colors}")
                            sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol, goal_heuristic=dist_heuristic)
                            if sol:
                                s._bfs_solution = sol
                                s._bfs_step = 0
                                return sol
            except Exception as e:
                logger.warning(f"BFS L{level_idx}: distance heuristic retry failed: {e}")
        
        return None

    def _tensor(s, fd):
        frame = s._raw(fd)
        oh=torch.zeros(16,64,64,dtype=torch.float32)
        oh.scatter_(0,torch.from_numpy(frame).unsqueeze(0),1)
        cnt=np.bincount(frame.flatten(),minlength=16)
        border=np.concatenate([frame[0,:],frame[-1,:],frame[:,0],frame[:,-1]])
        s._bg=int(np.bincount(border.flatten(),minlength=16).argmax());mx=max(cnt.max(),1)
        bg_m=(frame==s._bg).astype(np.float32)
        rar=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0:rar[frame==c]=1.0-cnt[c]/mx
        pad=np.pad(frame,1,mode='edge')
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|(frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        d1=torch.zeros(3,64,64,dtype=torch.float32)
        for i,prev in enumerate(reversed(list(s.fhist))):
            if i>=3:break
            d1[i]=torch.from_numpy((frame!=prev).astype(np.float32))
        d2=torch.zeros(2,64,64,dtype=torch.float32)
        h=list(s.fhist)
        if len(h)>=2:d2[0]=torch.from_numpy((h[-1]!=h[-2]).astype(np.float32))
        if len(h)>=4:d2[1]=torch.from_numpy((h[-2]!=h[-4]).astype(np.float32))
        s.fhist.append(frame.copy())
        return torch.cat([oh,aug,d1,d2],0).to(s.device)

    def _detect_template(s, frame):
        mask=torch.ones(4096,dtype=torch.float32)
        col_act=np.sum(frame!=s._bg,axis=0)
        for c in range(20,44):
            if col_act[c]<=2 and np.sum(col_act[:c]>0)>=5 and np.sum(col_act[c+1:]>0)>=5:
                for y in range(64):
                    for x in range(c+1):mask[y*64+x]=0.05
                return mask
        row_act=np.sum(frame!=s._bg,axis=1)
        for r in range(20,44):
            if row_act[r]<=2 and np.sum(row_act[:r]>0)>=5 and np.sum(row_act[r+1:]>0)>=5:
                for y in range(r+1):
                    for x in range(64):mask[y*64+x]=0.05
                return mask
        return mask

    def _reward(s, prev_raw, curr_raw, prev_h, curr_h):
        # FIX 1: Use s._visited_hashes (now properly initialized) for deduplication.
        # Previously _visited_hashes was never created, so the hasattr() check always
        # returned True (not hasattr = True) meaning every state change got +1.5,
        # causing the CNN to loop endlessly without penalty.
        mask=np.ones((64,64),dtype=bool);mask[:2]=False;mask[62:]=False
        diff=(prev_raw!=curr_raw)&mask;changed=np.any(diff)
        r=0.0
        if curr_h != prev_h:
            if curr_h not in s._visited_hashes:
                r += 1.5
                s._visited_hashes.add(curr_h)
            else:
                r += 0.2  # small reward for revisiting — not zero, avoids cliff in sparse games
        else:
            r -= 0.1
        if changed:r+=0.5
        curr_objs=fast_objects(curr_raw,s._bg)
        if s._prev_objs and curr_objs:
            moved=0
            for co in curr_objs:
                for po in s._prev_objs:
                    if co[0]==po[0]:
                        dist=abs(co[1]-po[1])+abs(co[2]-po[2])
                        if 2<dist<20:moved+=1;break
            if moved>0:r+=0.3*min(moved,3);s._obj_moved=moved
        s._prev_objs=curr_objs
        return r

    def _sample(s, logits, avail=None, temp=1.0):
        al=logits[:5].clone();cl=logits[5:5+4096].clone()
        if avail is not None and len(avail)>0:
            mask=torch.full_like(al,float('-inf'));a6=False
            for a in avail:
                aid=a.value if hasattr(a,'value') else int(a)
                if 1<=aid<=5:mask[aid-1]=0.0
                elif aid==6:a6=True
            al=al+mask
            if not a6:cl=cl+torch.full_like(cl,float('-inf'))
        if s._wm is not None:cl=cl+torch.log(s._wm.to(s.device).clamp(min=0.01))
        ap=torch.sigmoid(al/temp);cp=torch.sigmoid(cl/temp)/(s.G*s.G)
        allp=torch.cat([ap,cp]);sm=allp.sum()
        if sm<1e-8:allp=torch.ones_like(allp)/len(allp)
        else:allp=allp/sm
        idx=np.random.choice(len(allp),p=allp.cpu().numpy())
        if idx<5:return idx,None
        ci=idx-5;return 5,(ci//s.G,ci%s.G)

    def _heuristic(s, frame, avail, step):
        av=set(int(a.value) if hasattr(a,'value') else int(a) for a in avail)
        for d in[1,2,3,4]:
            if d in av and step<4:return d-1,None
        if 6 in av:
            cnt=np.bincount(frame.flatten(),minlength=16);targets=[]
            for c in range(16):
                if c==s._bg or cnt[c]==0 or cnt[c]>2000:continue
                ys,xs=np.where(frame==c)
                if len(ys)>=2:targets.append((int(np.median(xs)),int(np.median(ys)),len(ys)))
            targets.sort(key=lambda t:t[2]);pidx=step-4
            if 0<=pidx<len(targets):return 5,(targets[pidx][1],targets[pidx][0])
        if 5 in av:return 4,None
        choices=[a for a in av if 1<=a<=5]
        if choices:return random.choice(choices)-1,None
        return 0,None

    def _frame_to_tensor(s, frame):
        oh=torch.zeros(16,64,64,dtype=torch.float32)
        oh.scatter_(0,torch.from_numpy(frame).unsqueeze(0),1)
        cnt=np.bincount(frame.flatten(),minlength=16)
        bg=int(cnt.argmax());mx=max(cnt.max(),1)
        bg_m=(frame==bg).astype(np.float32)
        rar=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0:rar[frame==c]=1.0-cnt[c]/mx
        pad=np.pad(frame,1,mode='edge')
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|(frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        zeros=torch.zeros(5,64,64,dtype=torch.float32)
        return torch.cat([oh,aug,zeros],0)

    def _train(s):
        if len(s.buf)<s.bsz:return
        indices=np.random.choice(len(s.buf),s.bsz,replace=False)
        batch=[s.buf[i] for i in indices]
        states=torch.stack([s._frame_to_tensor(e['s']).to(s.device) for e in batch])
        acts=torch.tensor([e['a'] for e in batch],dtype=torch.long,device=s.device)
        rews=torch.tensor([e['r'] for e in batch],dtype=torch.float32,device=s.device)
        rews=torch.sigmoid(rews);s.opt.zero_grad()
        logits=s.net(states)
        acts_c=acts.clamp(0,logits.size(1)-1)
        sel=logits.gather(1,acts_c.unsqueeze(1)).squeeze(1)
        loss=F.binary_cross_entropy_with_logits(sel,rews)
        p=torch.sigmoid(logits);loss=loss-0.0001*p[:,:5].mean()-0.00001*p[:,5:].mean()
        loss.backward();s.opt.step()

    def _get_aem_tensors(s):
        if len(s._aem_diffs)<2:return None,None,None
        M=len(s._aem_diffs)
        diffs=torch.zeros(1,M,1,64,64,device=s.device)
        acts=torch.zeros(1,M,dtype=torch.long,device=s.device)
        rews=torch.zeros(1,M,device=s.device)
        for i,(d,a,r) in enumerate(zip(s._aem_diffs,s._aem_actions,s._aem_rewards)):
            diffs[0,i,0]=torch.from_numpy(d.astype(np.float32));acts[0,i]=min(a,4);rews[0,i]=r
        return diffs,acts,rews

    def is_done(s, frames, lf):
        try: return lf.state is GameState.WIN or (time.time()-s.start_time) >= 8*3600-300
        except: return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE =====
            if lvl != s.cl:
                # Init BFS solver on first level
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()

                # Try BFS for this level
                s._bfs_solution = None
                s._bfs_step = 0
                if s._bfs:
                    s._try_bfs_solve(lvl)

                # Init CNN fallback
                s.buf.clear(); s.buf_h.clear()
                s.net = ForgeNet(s.IN, s.G).to(s.device)
                for wp in ['/kaggle/input/forge-pretrained-weights/pretrained_weights.pt',
                           'pretrained_weights.pt']:
                    try:
                        if os.path.exists(wp):
                            state=torch.load(wp,map_location=s.device,weights_only=True)
                            ms=s.net.state_dict()
                            for k in list(state.keys()):
                                if k in ms and state[k].shape==ms[k].shape:ms[k]=state[k]
                            s.net.load_state_dict(ms);break
                    except: pass
                s.opt = optim.Adam(s.net.parameters(), lr=0.0003)
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
                s.cl=lvl;s.fhist.clear();s.la=0
                s._wd=False;s._wm=None
                s._aem_diffs.clear();s._aem_actions.clear();s._aem_rewards.clear()
                s._prev_objs=None;s._obj_moved=0;s._ckpt_hash=None;s._unproductive=0
                # FIX 1: Reset visited hashes on every level change
                s._visited_hashes = set()
                # FIX 4: Only reset epsilon if BFS didn't solve this level.
                # If BFS solved it, keep current eps so CNN fallback (if needed)
                # benefits from accumulated exploration knowledge.
                if not s._bfs_solution:
                    s._eps = 0.15

                # CLTI — inject BFS demos from previous level into CNN replay buffer
                # FIX 2: Use perform_action frame[-1] consistently with _raw(),
                # instead of get_pixels() which returns a different format.
                if lvl > 0 and s._bfs and s._bfs.solutions.get(lvl - 1):
                    prev_sol = s._bfs.solutions[lvl - 1]
                    try:
                        replay_game = s._bfs.game_cls()
                        replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        r0 = replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        if r0.frame:
                            # Start from the post-reset frame, consistent with _raw()
                            prev_frame = np.array(r0.frame[-1], dtype=np.int64)
                            for act_id, data in prev_sol:
                                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                result = replay_game.perform_action(ai, raw=True)
                                action_idx = (act_id - 1) if act_id <= 5 else (
                                    5 + data.get('y', 0) * 64 + data.get('x', 0) if data else 0)
                                s.buf.append({'s': prev_frame.copy(), 'a': action_idx, 'r': 2.0})
                                # Advance prev_frame using the action result, not get_pixels()
                                if result.frame:
                                    prev_frame = np.array(result.frame[-1], dtype=np.int64)
                            if len(s.buf) >= s.bsz:
                                for _ in range(min(20, len(s.buf) // s.bsz)):
                                    s._train()
                                logger.info(f"CLTI: injected {len(prev_sol)} expert demos from L{lvl-1}")
                    except Exception as e:
                        logger.warning(f"CLTI failed: {e}")

            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
                return GameAction.RESET

            # ===== BFS SOLUTION EXECUTION =====
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                s._last_action_data = {k: v for k, v in data.items() if k != 'game_id'} if data else None
                raw = s._raw(lf)
                s.fhist.append(raw.copy())
                s.pr = raw.copy()
                s.la += 1
                return sel

            # ===== CNN FALLBACK =====
            tensor = s._tensor(lf)
            raw = s._raw(lf)
            ch = hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail = getattr(lf, 'available_actions', None) or []
            s._undo_avail = any((a.value if hasattr(a,'value') else int(a))==7 for a in avail)

            if s.pt is not None and s.pai is not None:
                mask=np.ones((64,64),dtype=bool);mask[:2]=False;mask[62:]=False
                diff_map=(s.pr!=raw)&mask;changed=np.any(diff_map)
                eh=hashlib.md5(s.pr.tobytes()[:1000]+str(s.pai).encode()).hexdigest()[:16]
                if eh not in s.buf_h:
                    r=s._reward(s.pr,raw,'',ch)
                    s.buf.append({'s':s.pr.copy(),'a':s.pai,'r':r})
                    s.buf_h.add(eh)
                    if changed:
                        s._aem_diffs.append(diff_map)
                        s._aem_actions.append(min(s.pai,4))
                        s._aem_rewards.append(r)
                if changed:s._ckpt_hash=ch;s._unproductive=0
                else:s._unproductive+=1

            avail_idx=[]
            for a in avail:
                aid=a.value if hasattr(a,'value') else int(a)
                if 1<=aid<=5:avail_idx.append(aid-1)
                elif aid==6:avail_idx.extend([5+i for i in range(0,4096,128)])

            if s._wm is None:s._wm=s._detect_template(raw)

            if s._undo_avail and s._unproductive>=30 and s._ckpt_hash:
                s._unproductive=0;a=GameAction.ACTION7;a.reasoning="undo"
                s.pt=tensor;s.pai=6;s.pr=raw.copy();s.ph=ch;s.la+=1;return a

            if not s._wd:
                if s.la<10:aidx,coords=s._heuristic(raw,avail,s.la)
                else:
                    s._wd=True
                    for _ in range(min(5,len(s.buf)//s.bsz)):s._train()

            if s._wd:
                if random.random()<s._eps:
                    aidx,coords=s._sample(torch.zeros(4101,device=s.device),avail,temp=2.0)
                else:
                    with torch.no_grad():
                        mem=s._get_aem_tensors()
                        if mem[0] is not None:logits=s.net(tensor.unsqueeze(0),*mem).squeeze(0)
                        else:logits=s.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx,coords=s._sample(logits,avail,temp=0.5)
                s._eps=max(s._eps_min,s._eps*s._eps_decay)
            elif s.la>=10:s._wd=True;aidx,coords=0,None

            if aidx<5:
                sel=s.al[aidx]
            else:
                sel=GameAction.ACTION6;y,x=coords
                s._last_action_data={"x":int(x),"y":int(y)}
                
            # Analyse frame for reasoning log
            _cnt = np.bincount(raw.flatten(), minlength=16)
            _border = np.concatenate([raw[0,:], raw[-1,:], raw[:,0], raw[:,-1]])
            _bg = int(np.bincount(_border.flatten(), minlength=16).argmax())
            _total_px = raw.size
            _color_info = {}
            for _c in range(16):
                if _c == _bg or _cnt[_c] == 0: continue
                _ys, _xs = np.where(raw == _c)
                _color_info[_c] = {
                    "px": int(_cnt[_c]),
                    "cx": round(float(np.mean(_xs)), 1),
                    "cy": round(float(np.mean(_ys)), 1),
                }
            # Classify colors by pixel count: rare=player/target, common=wall/floor
            _sorted = sorted(_color_info.items(), key=lambda x: x[1]["px"])
            _rare = [c for c, _ in _sorted if _color_info[c]["px"] < _total_px * 0.05]
            _common = [c for c, _ in _sorted if _color_info[c]["px"] >= _total_px * 0.05]
            # Detect movement since last frame
            _moved_colors = []
            if s.pr is not None:
                for _c in _rare:
                    _prev_mask = (s.pr == _c)
                    _curr_mask = (raw == _c)
                    if np.any(_prev_mask) and np.any(_curr_mask):
                        _py, _px = np.where(_prev_mask)
                        _cy2, _cx2 = np.where(_curr_mask)
                        _dx = float(np.mean(_cx2)) - float(np.mean(_px))
                        _dy = float(np.mean(_cy2)) - float(np.mean(_py))
                        if abs(_dx) + abs(_dy) > 1:
                            _moved_colors.append({"c": _c, "dx": round(_dx,1), "dy": round(_dy,1)})

            sel.reasoning = {
                "step": s.la,
                "lvl": int(s.cl),
                "mode": "bfs" if (s._bfs_solution and s._bfs_step <= len(s._bfs_solution)) else ("exploit" if s._wd else "explore"),
                "act": f"A{aidx+1}" if aidx < 5 else f"click({int(coords[1])},{int(coords[0])})",
                "eps": round(s._eps, 3),
                "bg": _bg,
                "colors": {str(c): v for c, v in _color_info.items()},
                "rare": _rare,
                "common": _common,
                "moved": _moved_colors,
                "buf": len(s.buf),
            }
            
            s.pt=tensor;s.pai=aidx if aidx<5 else(5+coords[0]*s.G+coords[1])
            s.pr=raw.copy();s.ph=ch;s.la+=1
            if s.action_counter%s.tfreq==0 and s._wd:s._train()
            return sel

        except Exception as e:
            traceback.print_exc()
            a=random.choice(s.al);a.reasoning=f"err:{e}";return a