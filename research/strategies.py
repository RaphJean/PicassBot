import numpy as np
import copy
import torch
from picassbot.engine import DrawingWorld
from policy.model import PolicyNetwork

class SearchStrategy:
    def __init__(self, world_config, target_image, step_penalty=0.00001, action_scale=0.3, allow_early_stopping=True, policy_model_path=None):
        self.world_config = world_config
        self.target_image = target_image # Numpy array (H, W)
        self.height, self.width = target_image.shape
        self.step_penalty = step_penalty
        self.action_scale = action_scale
        self.allow_early_stopping = allow_early_stopping
        
        self.policy_model = None
        self.device = None
        if policy_model_path:
            try:
                # Determine device first
                self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
                # Load checkpoint (supports raw state_dict or full checkpoint dict)
                checkpoint = torch.load(policy_model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                # Infer hidden_dim from fusion layer shape
                fusion_weight = state_dict.get('fusion.0.weight')
                if fusion_weight is not None:
                    hidden_dim = fusion_weight.shape[0] // 2  # because first linear outputs hidden_dim*2
                else:
                    hidden_dim = 256  # fallback to default
                # Instantiate model with inferred hidden_dim
                self.policy_model = PolicyNetwork(hidden_dim=hidden_dim).to(self.device)
                self.policy_model.load_state_dict(state_dict)
                self.policy_model.eval()
                print(f"Loaded policy guidance from {policy_model_path} (hidden_dim={hidden_dim})")
            except Exception as e:
                print(f"Failed to load policy: {e}")
                self.policy_model = None

    def sample_action(self, state, step_idx, min_steps=5):
        """Sample an action from Policy if available, else Random."""
        if self.policy_model:
            with torch.no_grad():
                # Prepare inputs
                current_tensor = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0) / 255.0
                target_tensor = torch.from_numpy(self.target_image).float().unsqueeze(0).unsqueeze(0) / 255.0
                
                current_tensor = current_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                
                # Get action distribution
                action_tensor = self.policy_model.get_action(current_tensor, target_tensor, deterministic=False)
                action = action_tensor.cpu().numpy()[0] # (dx, dy, eos, eod)
                
                dx, dy, eos, eod = action
                
                # Enforce constraints
                if not self.allow_early_stopping or step_idx < min_steps:
                    eod = 0.0
                else:
                    # Policy outputs 0 or 1 for EOD, we keep it
                    pass
                    
                return (dx, dy, eos, eod)
        else:
            # Fallback to random
            dx = np.random.uniform(-self.action_scale, self.action_scale)
            dy = np.random.uniform(-self.action_scale, self.action_scale)
            eos = 1 if np.random.random() < 0.2 else 0
            
            if self.allow_early_stopping and step_idx >= min_steps:
                eod = 1 if np.random.random() < 0.05 else 0
            else:
                eod = 0
            
            return (dx, dy, eos, eod)

    def loss(self, image):
        """Calculate Mean Squared Error between blurred image and target."""
        from scipy.ndimage import gaussian_filter
        
        # Normalize to [0, 1] and cast to float
        img_norm = image.astype(np.float32) / 255.0
        target_norm = self.target_image.astype(np.float32) / 255.0
        
        # Apply Gaussian Blur to smooth the loss landscape
        sigma = 2.0
        img_blur = gaussian_filter(img_norm, sigma=sigma)
        target_blur = gaussian_filter(target_norm, sigma=sigma)
        
        return np.mean((img_blur - target_blur) ** 2)

    def run(self, max_steps=50):
        raise NotImplementedError

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


class LatentMPC(SearchStrategy):
    """
    Model Predictive Control in latent space using FullAgent.
    
    Uses FullAgent's learned encoder, predictor, and policy for planning.
    Requires: joint_model_path (checkpoint from FullAgent training)
    """
    
    def __init__(self, world_config, target_image, step_penalty=0.00001, 
                 joint_model_path=None, **kwargs):
        # Don't pass policy_model_path to parent - we'll handle loading ourselves
        super().__init__(world_config, target_image, step_penalty=step_penalty, 
                        policy_model_path=None, **kwargs)
        
        if not joint_model_path:
            raise ValueError("LatentMPC requires joint_model_path (FullAgent checkpoint)")
        
        # Import FullAgent
        from policy.joint_model import FullAgent
        
        try:
            # Determine device
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() 
                else "cuda" if torch.cuda.is_available() 
                else "cpu"
            )
            
            # Load checkpoint
            checkpoint = torch.load(joint_model_path, map_location=self.device)
            
            # Get config from checkpoint or use defaults
            if 'config' in checkpoint:
                cfg = checkpoint['config']
                action_dim = cfg['model']['action_dim']
                hidden_dim = cfg['model']['hidden_dim']
            else:
                action_dim = 4
                hidden_dim = 512
            
            # Create FullAgent model
            self.full_agent = FullAgent(
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ).to(self.device)
            
            # Load state dicts for each component
            self.full_agent.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.full_agent.predictor.load_state_dict(checkpoint['predictor_state_dict'])
            self.full_agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            
            self.full_agent.eval()
            print(f"Loaded FullAgent from {joint_model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load FullAgent: {e}")
        
        # Encode target image once
        with torch.no_grad():
            target_tensor = torch.from_numpy(target_image).float().unsqueeze(0).unsqueeze(0) / 255.0
            target_tensor = target_tensor.to(self.device)
            self.z_target = self.full_agent.encoder(target_tensor)
    
    def encode_state(self, state):
        """Encode canvas state to latent representation."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0) / 255.0
            state_tensor = state_tensor.to(self.device)
            z = self.full_agent.encoder(state_tensor)
        return z
    
    def generate_action_from_latent(self, z_t, deterministic=False):
        """Generate action from latent state using FullAgent policy."""
        with torch.no_grad():
            mean, logstd, eos_logit, eod_logit = self.full_agent.policy(z_t, self.z_target)
            
            logstd = torch.clamp(logstd, min=-5, max=2)
            std = torch.exp(logstd)
            
            if deterministic:
                dx_dy = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                dx_dy = dist.sample()
            
            eos_prob = torch.sigmoid(eos_logit)
            eod_prob = torch.sigmoid(eod_logit)
            
            if deterministic:
                eos = (eos_prob > 0.5).float()
                eod = (eod_prob > 0.5).float()
            else:
                eos = torch.bernoulli(eos_prob)
                eod = torch.bernoulli(eod_prob)
            
            action = torch.cat([dx_dy, eos, eod], dim=1).squeeze(0).cpu().numpy()
        
        return action
    
    def latent_loss(self, z_t):
        """Compute loss in latent space (L2 distance to target)."""
        loss = torch.nn.functional.mse_loss(z_t, self.z_target)
        return loss.item()
    
    def run(self, max_steps=50, horizon=5, num_sequences=20):
        """Run Latent MPC with FullAgent."""
        world = DrawingWorld(**self.world_config)
        history = []
        
        for step in range(max_steps):
            z_t = self.encode_state(world.get_state())
            
            best_sequence = None
            best_cost = float('inf')
            
            # Evaluate sequences in latent space
            for _ in range(num_sequences):
                z_sim = z_t.clone()
                sequence = []
                seq_cost = float('inf')
                
                for t in range(horizon):
                    action = self.generate_action_from_latent(z_sim, deterministic=False)
                    dx, dy, eos, eod = action
                    sequence.append(action)
                    
                    if eod > 0.5:
                        loss = self.latent_loss(z_sim)
                        seq_cost = loss + self.step_penalty * t
                        break
                    
                    # Predict next latent state using FullAgent predictor
                    action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
                    z_next, _ = self.full_agent.predictor(z_sim, action_tensor)
                    z_sim = z_next
                
                if seq_cost == float('inf'):
                    loss = self.latent_loss(z_sim)
                    seq_cost = loss + self.step_penalty * horizon
                
                if seq_cost < best_cost:
                    best_cost = seq_cost
                    best_sequence = sequence
            
            # Apply first action in real world
            if best_sequence:
                first_action = best_sequence[0]
                dx, dy, eos, eod = first_action
                
                if eod > 0.5:
                    print(f"Step {step}: Stopping (EOD). Latent Cost {best_cost:.4f}")
                    break
                
                world.step(first_action)
                history.append(world.get_state())
                print(f"Step {step}: Latent Cost {best_cost:.4f}")
            else:
                break
        
        return history

