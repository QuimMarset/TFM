import torch as th
from learners.facmac_learner import FACMACLearner



class FACMACDistributionLearner(FACMACLearner):

    def compute_actor_loss(self, batch, t_env):
        qs = []
        log_probs = []

        self.actor.init_hidden(batch.batch_size)
        self.critic.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            actions_t, log_probs_t = self.actor.select_actions_with_log_probs(batch, t_ep=t)
            ind_qs = self.critic.forward(batch, t, actions_t)
            joint_qs = self.mixer(ind_qs, batch["state"][:, t:t+1])
            
            log_probs.append(log_probs_t)
            qs.append(joint_qs)
        
        log_probs = th.stack(log_probs[:-1], dim=1)
        qs = th.stack(qs[:-1], dim=1)
        loss = - qs.mean() + (log_probs**2).mean() * 1e-3

        actor_metrics = {
            'actor_loss' : loss.item(),
            'actor_qs_mean' : qs.mean().item()
        }
        return loss, actor_metrics
    

    def _compute_target_actions(self, batch, t_env):
        target_actions = []
        self.target_actor.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            # Set test_mode to False in this version with distributions
            target_actions_t = self.target_actor.select_actions(batch, t_ep=t, t_env=t_env)
            target_actions.append(target_actions_t)
        
        return th.stack(target_actions, dim=1)