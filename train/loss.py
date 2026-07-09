"""
==============================================================================
File: loss.py
Description: 
    Custom loss functions for survival analysis models.
    This module implements the Cox Negative Log Partial Likelihood loss 
    function with built-in L2 Regularization to prevent overfitting in 
    high-dimensional multi-omics datasets.
==============================================================================
"""

import torch
import torch.nn as nn

class Regularization(object):
    """
    Custom Regularization Class.
    
    This class iterates through all the parameters of a PyTorch model and 
    computes the Lp-norm (e.g., L1 or L2 norm) specifically for the 'weight' 
    parameters. It ignores biases to prevent underfitting.
    """
    def __init__(self, order, weight_decay):
        """
        Initialize the regularization function.
        
        Args:
            order (int): The order of the norm (e.g., 1 for L1, 2 for L2).
            weight_decay (float): The regularization strength (lambda parameter).
        """
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        """
        Compute the total regularization penalty for the model.
        
        Args:
            model (nn.Module): The PyTorch model to be regularized.
            
        Returns:
            torch.Tensor: The computed regularization loss.
        """
        reg_loss = 0
        for name, w in model.named_parameters():
            # Apply regularization ONLY to weights, not to biases
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        return self.weight_decay * reg_loss


class NegativeLogLikelihood(nn.Module):
    """
    Cox Proportional Hazards Negative Log Partial Likelihood Loss.
    
    This loss function evaluates the ranking of predicted risk scores 
    against the actual survival times, heavily penalizing models that 
    assign lower risk to patients who experienced an event earlier.
    It includes numerical stability tricks (log-sum-exp) and L2 regularization.
    """
    def __init__(self, config):
        """
        Initialize the Cox loss module.
        
        Args:
            config (dict): Configuration dictionary containing 'l2_reg' strength.
        """
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = config['l2_reg']
        # Instantiate L2 regularization (order=2)
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)
        # Small epsilon value to prevent log(0) errors during calculation
        self.eps = 1e-7

    def forward(self, risk_pred, y, e, model):
        """
        Forward pass to compute the total survival loss.
        
        Args:
            risk_pred (torch.Tensor): Predicted hazard/risk scores [batch_size, 1].
            y (torch.Tensor): Actual survival times or follow-up durations [batch_size].
            e (torch.Tensor): Event indicators (1 if event occurred, 0 if censored) [batch_size].
            model (nn.Module): The active PyTorch model (passed to compute weight penalty).
            
        Returns:
            torch.Tensor: The combined Cox loss and L2 penalty.
        """
        # Flatten tensors to 1D to ensure correct broadcasting and indexing
        risk_pred = risk_pred.view(-1)
        y = y.view(-1)
        e = e.view(-1)
        
        # Step 1: Sort all patients by survival time in descending order
        # Patients who survived the longest are placed first.
        order = torch.argsort(y, descending=True)
        sorted_risk = risk_pred[order]
        sorted_y = y[order]
        sorted_e = e[order]
        
        n = len(y)
        
        # Step 2: Determine the Risk Set for each patient
        # R(t_i) includes all patients whose survival time is >= patient i's survival time.
        # Since the data is sorted descendingly, the risk set for patient `i` 
        # is simply all patients from index `0` up to `i`.
        risk_sets = [(sorted_y >= sorted_y[i]) for i in range(n)]
        
        log_sum_exps = []
        
        # Step 3: Compute the Log-Sum-Exp for each patient who experienced an event
        for i in range(n):
            if sorted_e[i] > 0.5:  # If the event occurred (uncensored)
                R = risk_sets[i]
                
                # --- Numerical Stability Trick (Log-Sum-Exp) ---
                # To prevent exponential overflow, we subtract the maximum risk score 
                # in the current risk set before exponentiating, and add it back outside.
                max_risk = torch.max(sorted_risk[R])
                exp_values = torch.exp(sorted_risk[R] - max_risk)
                log_sum_exp = max_risk + torch.log(torch.sum(exp_values) + self.eps)
                
                log_sum_exps.append(log_sum_exp)
        
        # Create a boolean mask for actual events
        event_mask = (sorted_e > 0.5)
        n_events = torch.sum(event_mask).item()
        
        # Step 4: Aggregate the Negative Log Partial Likelihood
        if n_events > 0:
            event_log_sum_exps = torch.stack(log_sum_exps)
            event_risks = sorted_risk[event_mask]
            
            # Cox Loss formula: -1/N * Sum(risk_i - log(Sum(exp(risk_j for j in R_i))))
            cox_loss = torch.sum(event_log_sum_exps - event_risks) / n_events
        else:
            # Handle edge case where a batch has absolutely no events (all censored)
            cox_loss = torch.tensor(0.0, device=y.device)
            
        # Step 5: Add the L2 weight decay penalty and return
        return cox_loss + self.reg(model)