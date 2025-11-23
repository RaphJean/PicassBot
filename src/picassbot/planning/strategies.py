import numpy as np
import copy
import torch
from picassbot.engine import DrawingWorld
from picassbot.policy.model import PolicyNetwork
from picassbot.planning.latent_mpc import LatentMPC

from picassbot.planning.base import SearchStrategy

class GreedySearch(SearchStrategy):
    def run(self, max_steps=50, samples_per_step=50):
        world = DrawingWorld(**self.world_config)
        history = []
        
        for step in range(max_steps):
            best_action = None
            best_cost = float('inf')
            
            # Try N random actions
            for _ in range(samples_per_step):
                # Random action: dx, dy in [-scale, scale], eos/eod
                dx = np.random.uniform(-self.action_scale, self.action_scale)
                dy = np.random.uniform(-self.action_scale, self.action_scale)
                eos = 1 if np.random.random() < 0.2 else 0 # 20% chance to lift pen
                
                # Small chance to stop, ONLY if step >= min_steps AND early stopping is allowed
                min_steps = 5
                if self.allow_early_stopping and step >= min_steps:
                    eod = 1 if np.random.random() < 0.05 else 0
                else:
                    eod = 0
                
                action = (dx, dy, eos, eod)
                
                # Simulate
                if eod:
                    # If we stop, the state doesn't change, but we pay no more penalties for future steps?
                    # Actually, if we stop now, the cost is just the current loss.
                    # We compare this against continuing (which incurs penalty).
                    current_loss = self.loss(world.get_state())
                    cost = current_loss # No step penalty for stopping action itself? Or maybe yes?
                    # Let's say stopping is an action, so it costs 1 penalty, but then 0 for future.
                    # Ideally we compare: Loss_now vs (Loss_next + penalty)
                else:
                    temp_world = world.copy()
                    temp_world.step(action)
                    current_loss = self.loss(temp_world.get_state())
                    cost = current_loss + self.step_penalty
                
                if cost < best_cost:
                    best_cost = cost
                    best_action = action
            
            # Apply best action
            if best_action:
                dx, dy, eos, eod = best_action
                if eod:
                    print(f"Step {step}: Stopping (EOD selected). Cost {best_cost:.4f}")
                    break
                
                world.step(best_action)
                history.append(world.get_state())
                print(f"Step {step}: Cost {best_cost:.4f}")
            else:
                break
                
        return history

class RandomShootingMPC(SearchStrategy):
    def run(self, max_steps=50, horizon=5, num_sequences=20):
        world = DrawingWorld(**self.world_config)
        history = []
        
        for step in range(max_steps):
            best_sequence = None
            best_cost = float('inf')
            
            # Generate random sequences
            for _ in range(num_sequences):
                temp_world = world.copy()
                sequence = []
                seq_cost = float('inf')
                
                for t in range(horizon):
                    min_steps = 5
                    action = self.sample_action(temp_world.get_state(), step + t, min_steps=min_steps)
                    dx, dy, eos, eod = action
                    sequence.append(action)
                    
                    if eod:
                        # Stop here
                        loss = self.loss(temp_world.get_state())
                        # Cost = Loss + penalty * steps_taken
                        # steps_taken is t (0-indexed, so t actions before this? No, t+1 actions including this one)
                        # Actually if EOD is immediate, we stop.
                        # Let's say cost is loss at stop time + penalty * (t)
                        seq_cost = loss + self.step_penalty * t
                        break
                    
                    temp_world.step(action)
                
                if seq_cost == float('inf'):
                    # Did not stop within horizon
                    loss = self.loss(temp_world.get_state())
                    seq_cost = loss + self.step_penalty * horizon
                
                if seq_cost < best_cost:
                    best_cost = seq_cost
                    best_sequence = sequence
            
            # Apply ONLY the first action of the best sequence (MPC principle)
            if best_sequence:
                first_action = best_sequence[0]
                dx, dy, eos, eod = first_action
                
                if eod:
                    print(f"Step {step}: Stopping (EOD selected). Cost {best_cost:.4f}")
                    break

                world.step(first_action)
                history.append(world.get_state())
                print(f"Step {step}: Forecast Cost {best_cost:.4f}")
            else:
                break
                
        return history

class GeneticSearch(SearchStrategy):
    def run(self, max_steps=50, population_size=20, generations=5):
        """
        A simplified evolutionary approach.
        At each step, evolve a population of actions to find the best next move.
        """
        world = DrawingWorld(**self.world_config)
        history = []
        
        for step in range(max_steps):
            # Initialize population of actions
            population = []
            min_steps = 5
            for _ in range(population_size):
                dx = np.random.uniform(-self.action_scale, self.action_scale)
                dy = np.random.uniform(-self.action_scale, self.action_scale)
                eos = 1 if np.random.random() < 0.2 else 0
                
                if self.allow_early_stopping and step >= min_steps:
                    eod = 1 if np.random.random() < 0.05 else 0
                else:
                    eod = 0
                    
                population.append((dx, dy, eos, eod))
            
            best_action = None
            best_cost = float('inf')

            # Evolve
            for gen in range(generations):
                # Evaluate
                scores = []
                for action in population:
                    dx, dy, eos, eod = action
                    if eod:
                        cost = self.loss(world.get_state())
                    else:
                        temp_world = world.copy()
                        temp_world.step(action)
                        loss = self.loss(temp_world.get_state())
                        cost = loss + self.step_penalty
                    
                    scores.append((cost, action))
                
                scores.sort(key=lambda x: x[0])
                best_cost = scores[0][0]
                best_action = scores[0][1]
                
                # Selection (Keep top 20%)
                top_k = int(population_size * 0.2)
                survivors = [s[1] for s in scores[:top_k]]
                
                # Mutation / Reproduction
                new_population = survivors[:]
                while len(new_population) < population_size:
                    parent = survivors[np.random.randint(len(survivors))]
                    # Mutate
                    dx = parent[0] + np.random.normal(0, 0.05)
                    dy = parent[1] + np.random.normal(0, 0.05)
                    eos = parent[2] 
                    eod = parent[3]
                    
                    # Mutate discrete flags
                    if np.random.random() < 0.1:
                        eos = 1 - eos
                    if np.random.random() < 0.05 and step >= min_steps:
                        eod = 1 - eod
                    elif step < min_steps:
                        eod = 0
                        
                    new_population.append((dx, dy, eos, eod))
                
                population = new_population

            # Apply best action found after generations
            dx, dy, eos, eod = best_action
            if eod:
                print(f"Step {step}: Stopping (EOD selected). Cost {best_cost:.4f}")
                break
                
            world.step(best_action)
            history.append(world.get_state())
            print(f"Step {step}: Cost {best_cost:.4f}")

        return history

class CEMSearch(SearchStrategy):
    def run(self, max_steps=50, horizon=10, num_sequences=50, num_elites=10, num_iterations=5, initial_std=0.5):
        world = DrawingWorld(**self.world_config)
        history = []
        
        for step in range(max_steps):
            # Initialize distribution parameters for the sequence of actions
            # Each action has 4 components: dx, dy, eos, eod
            
            # Mean and Std for (dx, dy, eos_prob, eod_prob) over the horizon
            # Shape: (horizon, 4)
            mean = np.zeros((horizon, 4))
            std = np.ones((horizon, 4)) * initial_std # Initial variance
            
            # For eos/eod, we might want to initialize mean such that eos is low (drawing)
            mean[:, 2] = 0.1 # Low prob of lifting
            mean[:, 3] = 0.0 # Low prob of ending
            
            best_sequence_overall = None
            best_cost_overall = float('inf')

            for iteration in range(num_iterations):
                sequences = []
                scores = []
                
                # Sample sequences
                for _ in range(num_sequences):
                    # Sample from Gaussian
                    sample = np.random.normal(mean, std)
                    
                    # Clip/Process actions
                    sample[:, 0:2] = np.clip(sample[:, 0:2], -1.0, 1.0)
                    
                    sequences.append(sample)
                    
                    # Evaluate
                    temp_world = world.copy()
                    seq_cost = float('inf')
                    
                    for t in range(horizon):
                        # Convert continuous sample to discrete action for environment
                        dx, dy = sample[t, 0], sample[t, 1]
                        eos = 1.0 if sample[t, 2] > 0.5 else 0.0
                        
                        min_steps = 5
                        if self.allow_early_stopping and (step + t) >= min_steps:
                            eod = 1.0 if sample[t, 3] > 0.5 else 0.0
                        else:
                            eod = 0.0
                        
                        if eod > 0.5:
                            # Stop here
                            loss = self.loss(temp_world.get_state())
                            seq_cost = loss + self.step_penalty * t
                            break
                            
                        temp_world.step((dx, dy, eos, eod))
                    
                    if seq_cost == float('inf'):
                        loss = self.loss(temp_world.get_state())
                        seq_cost = loss + self.step_penalty * horizon
                    
                    scores.append(seq_cost)
                    
                    if seq_cost < best_cost_overall:
                        best_cost_overall = seq_cost
                        best_sequence_overall = sample

                # Select elites
                elite_indices = np.argsort(scores)[:num_elites]
                elites = np.array([sequences[i] for i in elite_indices])
                
                # Update distribution
                new_mean = np.mean(elites, axis=0)
                new_std = np.std(elites, axis=0) + 1e-5 # Add epsilon
                
                # Soft update (optional, but good for stability)
                alpha = 0.5
                mean = alpha * new_mean + (1 - alpha) * mean
                std = alpha * new_std + (1 - alpha) * std

            # Execute first action of the best mean (or best sequence found)
            best_action_cont = best_sequence_overall[0]
            
            dx, dy = best_action_cont[0], best_action_cont[1]
            eos = 1.0 if best_action_cont[2] > 0.5 else 0.0
            
            min_steps = 5
            if self.allow_early_stopping and step >= min_steps:
                eod = 1.0 if best_action_cont[3] > 0.5 else 0.0
            else:
                eod = 0.0
            
            if eod > 0.5:
                print(f"Step {step}: Stopping (EOD selected). Cost {best_cost_overall:.4f}")
                break
            
            action = (dx, dy, eos, eod)
            world.step(action)
            history.append(world.get_state())
            print(f"Step {step}: Forecast Cost {best_cost_overall:.4f}")

        return history

class MCTSSearch(SearchStrategy):
    """Improved Monte Carlo Tree Search for drawing actions.
    Uses UCB (Upper Confidence Bound) for action selection, maintains statistics
    for each action tried, and progressively widens the action space.
    """
    def __init__(self, world_config, target_image, step_penalty=0.00001, action_scale=0.3, allow_early_stopping=True, policy_model_path=None):
        super().__init__(world_config, target_image, step_penalty=step_penalty, action_scale=action_scale, allow_early_stopping=allow_early_stopping, policy_model_path=policy_model_path)

    def run(self, max_steps=30, horizon=5, num_simulations=100, exploration_const=1.4):
        """Run improved MCTS search.
        Args:
            max_steps: maximum number of actions to take.
            horizon: length of each rollout.
            num_simulations: number of rollouts per step.
            exploration_const: UCB exploration constant (higher = more exploration).
        Returns:
            List of canvas states representing the drawing trajectory.
        """
        world = DrawingWorld(**self.world_config)
        history = []
        min_steps = 5
        
        for step in range(max_steps):
            # Action statistics: {action_tuple: {'visits': int, 'total_cost': float, 'best_cost': float}}
            action_stats = {}
            total_visits = 0
            
            # Run simulations
            for sim_idx in range(num_simulations):
                sim_world = world.copy()
                cumulative_cost = 0.0
                first_action = None
                
                # First action: use UCB if we have statistics, otherwise sample randomly
                if action_stats and sim_idx > 10:  # Start using UCB after initial exploration
                    # UCB selection for first action
                    best_ucb = -float('inf')
                    selected_action = None
                    
                    for action, stats in action_stats.items():
                        # UCB1 formula: avg_cost - exploration_bonus (we minimize cost)
                        avg_cost = stats['total_cost'] / stats['visits']
                        exploration_bonus = exploration_const * np.sqrt(np.log(total_visits) / stats['visits'])
                        ucb_value = -avg_cost + exploration_bonus  # Negative because we minimize cost
                        
                        if ucb_value > best_ucb:
                            best_ucb = ucb_value
                            selected_action = action
                    
                    first_action = selected_action
                else:
                    # Random sampling (or Policy sampling) for exploration
                    first_action = self.sample_action(sim_world.get_state(), step, min_steps=min_steps)
                
                # Execute first action
                sim_world.step(first_action)
                loss = self.loss(sim_world.get_state())
                cumulative_cost += loss + self.step_penalty
                
                # Check if first action is EOD
                if first_action[3]:  # eod flag
                    total_cost = cumulative_cost
                else:
                    # Continue rollout with random actions
                    for t in range(1, horizon):
                        action = self.sample_action(sim_world.get_state(), step + t, min_steps=min_steps)
                        dx, dy, eos, eod = action
                        sim_world.step(action)
                        loss = self.loss(sim_world.get_state())
                        cumulative_cost += loss + self.step_penalty
                        
                        if eod:
                            break
                    
                    total_cost = cumulative_cost
                
                # Update statistics for first action
                if first_action not in action_stats:
                    action_stats[first_action] = {
                        'visits': 0,
                        'total_cost': 0.0,
                        'best_cost': float('inf')
                    }
                
                action_stats[first_action]['visits'] += 1
                action_stats[first_action]['total_cost'] += total_cost
                action_stats[first_action]['best_cost'] = min(action_stats[first_action]['best_cost'], total_cost)
                total_visits += 1
            
            # Select best action based on lowest average cost
            best_action = None
            best_avg_cost = float('inf')
            
            for action, stats in action_stats.items():
                avg_cost = stats['total_cost'] / stats['visits']
                if avg_cost < best_avg_cost:
                    best_avg_cost = avg_cost
                    best_action = action
            
            # Apply selected action
            if best_action is None:
                break
            
            dx, dy, eos, eod = best_action
            if eod:
                print(f"Step {step}: Stopping (EOD selected). Cost {best_avg_cost:.4f}")
                break
            
            world.step(best_action)
            history.append(world.get_state())
            print(f"Step {step}: Cost {best_avg_cost:.4f} (from {len(action_stats)} actions, {total_visits} sims)")
        
        return history

class PolicyStrategy(SearchStrategy):
    def __init__(self, world_config, target_image, model_path, step_penalty=0.00001, action_scale=0.5, allow_early_stopping=True):
        super().__init__(world_config, target_image, step_penalty, action_scale, allow_early_stopping)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyNetwork().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded policy from {model_path}")
        except Exception as e:
            print(f"Error loading policy: {e}")
            
        self.model.eval()
        
    def run(self, max_steps=30, deterministic=False):
        world = DrawingWorld(**self.world_config)
        history = []
        
        with torch.no_grad():
            for step in range(max_steps):
                # Prepare inputs
                current_state = world.get_state()
                # (H, W) -> (1, 1, H, W)
                current_tensor = torch.from_numpy(current_state).float().unsqueeze(0).unsqueeze(0) / 255.0
                target_tensor = torch.from_numpy(self.target_image).float().unsqueeze(0).unsqueeze(0) / 255.0
                
                current_tensor = current_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                
                # Get action from policy
                action_tensor = self.model.get_action(current_tensor, target_tensor, deterministic=deterministic)
                action = action_tensor.cpu().numpy()[0] # (dx, dy, eos, eod)
                
                # Apply action
                dx, dy, eos, eod = action
                
                # Check EOD
                min_steps = 5
                if self.allow_early_stopping and step >= min_steps and eod > 0.5:
                    print(f"Step {step}: Stopping (Policy EOD).")
                    break
                
                # Force EOD to 0 if not stopping
                action_to_step = (dx, dy, eos, 0.0)
                
                world.step(action_to_step)
                history.append(world.get_state())
                
                loss = self.loss(world.get_state())
                print(f"Step {step}: Loss {loss:.4f}")
                
        return history




