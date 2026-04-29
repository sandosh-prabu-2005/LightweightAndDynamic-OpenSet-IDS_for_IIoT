"""
Custom optimizer implementations for IDS training.

This module contains:
- ABAO_V1: Original Adaptive Boundary Aware Optimizer (reference)
- ABAO_V2: Improved EMA-normalized ABAO (recommended)
- CRAZYFOX: Open-set IDS optimizer with recall and EVT-aware dynamics
- ABAOPlus: Stability-Aware Adaptive Optimizer with multi-loss weighting
"""

import torch
import torch.optim as optim
import numpy as np


class ABAO_V1(optim.Optimizer):
    """
    ABAO V1 - Original Adaptive Boundary Aware Optimizer.
    
    Note: This optimizer has a known issue where W_t collapses to its minimum (0.5),
    resulting in updates 50% weaker than Adam. Use ABAO_V2 instead for better performance.
    
    Formula: W_t = α·L_cls + β·L_rec + γ·(1−Conf)  |  W_clip=[0.5, 3.0]
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 alpha=0.4, beta_abao=0.3, gamma=0.3, w_min=0.5, w_max=3.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        alpha=alpha, beta_abao=beta_abao, gamma=gamma,
                        w_min=w_min, w_max=w_max)
        super().__init__(params, defaults)
        self._boundary_weights = []

    @property
    def boundary_weight_history(self):
        return self._boundary_weights

    @torch.no_grad()
    def step(self, loss_cls=0.0, loss_rec=0.0, conf=0.5, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lc = float(loss_cls) * 1.2
        lr_val = float(loss_rec)
        c = float(conf)

        for group in self.param_groups:
            alpha = group['alpha']
            beta_abao = group['beta_abao']
            gamma = group['gamma']
            lr = group['lr']
            b1, b2 = group['betas']
            eps_adam = group['eps']
            w_min = group['w_min']
            w_max = group['w_max']

            Wt = alpha * lc + beta_abao * lr_val + gamma * (1.0 - c)
            Wt = float(max(w_min, min(w_max, Wt)))

            self._boundary_weights.append(Wt)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                exp_avg.mul_(b1).add_(grad, alpha=1 - b1)
                exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1 - b2)

                m_hat = exp_avg / (1 - b1**t)
                v_hat = exp_avg_sq / (1 - b2**t)

                denom = v_hat.sqrt().add_(eps_adam)
                p.addcdiv_(m_hat, denom, value=-lr * Wt)

        return loss


class ABAO_V2(optim.Optimizer):
    """
    ABAO V2 - Improved Adaptive Boundary Aware Optimizer.
    
    ROOT CAUSE FIX over V1:
    - V1 W_t could collapse to 0.5 minimum, making updates 50% weaker than Adam
    - V2 uses EMA-normalized loss ratios with W_min=1.0, never underperforming Adam
    
    Formula:
    W_t = 1.0 + α·max(0, L_cls/EMA_cls−1) + β·max(0, L_rec/EMA_rec−1) + γ·(1−Conf)^τ
    
    Key improvements:
    - W_min=1.0 → ABAO never weaker than Adam
    - EMA normalization → responds to relative difficulty
    - Decoupled weight decay (AdamW-style)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-4,
                 alpha=0.6, beta_abao=0.4, gamma=0.6,
                 tau=0.3, ema_decay=0.95,
                 w_min=1.2, w_max=5.0):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        alpha=alpha, beta_abao=beta_abao,
                        gamma=gamma, tau=tau,
                        ema_decay=ema_decay,
                        w_min=w_min, w_max=w_max)

        super().__init__(params, defaults)

        self._ema_cls = None
        self._ema_rec = None
        self._boundary_weights = []

    @property
    def boundary_weight_history(self):
        return self._boundary_weights

    @torch.no_grad()
    def step(self, loss_cls=0.0, loss_rec=0.0, conf=0.5, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lc = float(loss_cls) * 1.2
        lr_val = float(loss_rec)
        c = float(conf)

        for group in self.param_groups:
            ema_decay = group['ema_decay']

        # EMA update
        if self._ema_cls is None:
            self._ema_cls = lc if lc > 1e-6 else 1.0
            self._ema_rec = lr_val if lr_val > 1e-6 else 1.0
        else:
            self._ema_cls = ema_decay * self._ema_cls + (1 - ema_decay) * lc
            self._ema_rec = ema_decay * self._ema_rec + (1 - ema_decay) * lr_val

        for group in self.param_groups:
            alpha = group['alpha']
            beta_a = group['beta_abao']
            gamma = group['gamma']
            tau = group['tau']
            lr = group['lr']
            wd = group['weight_decay']
            b1, b2 = group['betas']
            eps_adam = group['eps']
            w_min = group['w_min']
            w_max = group['w_max']

            # Relative difficulty
            cls_ratio = lc / (self._ema_cls + 1e-9)
            rec_ratio = lr_val / (self._ema_rec + 1e-9)

            w_cls = alpha * max(0.0, cls_ratio - 1.0)
            w_rec = beta_a * max(0.0, rec_ratio - 1.0)

            # Boundary signal
            w_conf = gamma * ((1.0 - c) ** tau)
            w_over = gamma * (c ** 2)

            # Final weight (stronger than Adam)
            Wt = 1.2 + w_cls + w_rec + w_conf + w_over
            Wt = float(max(w_min, min(w_max, Wt)))

            self._boundary_weights.append(Wt)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # AdamW decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                exp_avg.mul_(b1).add_(grad, alpha=1 - b1)
                exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1 - b2)

                m_hat = exp_avg / (1 - b1**t)
                v_hat = exp_avg_sq / (1 - b2**t)

                denom = v_hat.sqrt().add_(eps_adam)

                p.addcdiv_(m_hat, denom, value=-lr * Wt)

        return loss


class CRAZYFOX(optim.Optimizer):
    """
    CRAZYFOX - Open-set aware optimizer for IDS.

    This optimizer preserves Adam moments, EMA smoothing, gradient clipping,
    and decoupled weight decay while adding recall-sensitive scaling,
    confidence temperature scaling, and late-stage learning boosts.

    Main mechanisms:
    - Hard sample reinforcement for recall improvement.
    - Confidence temperature scaling to prevent overconfidence.
    - Late-stage LR boost to escape local minima.
    - Dynamic loss-aware scaling for IDS-oriented objective.
    - Decoupled AdamW-style weight decay.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-4,
                 grad_var_beta=0.95, grad_boost_alpha=0.45,
                 grad_boost_max=1.8, variance_floor=0.08,
                 conf_threshold=0.90, conf_beta=0.65,
                 loss_ema_beta=0.90, loss_temperature=0.70,
                 loss_boost=0.25, gamma_plateau=0.20,
                 plateau_beta=0.92, plateau_threshold=2e-3,
                 recall_alpha=0.35, hard_beta=0.15,
                 hard_conf_threshold=0.60, max_recall_scale=1.45,
                 class_ema_beta=0.98, class_boost_alpha=0.25,
                 class_boost_max=1.35,
                 confidence_gamma=0.75, min_conf_scale=0.85,
                 evt_margin=0.1, evt_lambda_start=0.05, evt_lambda_end=0.40,
                 loss_alpha=0.45, loss_beta=0.25, kl_weight=1.0,
                 sep_gamma=0.20, lambda_sep=0.03, lambda_dist=0.04,
                 curvature_mix=0.15, lr_min_scale=0.60, lr_max_scale=2.25):

        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= curvature_mix <= 1.0:
            raise ValueError("curvature_mix must be in [0, 1]")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_var_beta=grad_var_beta,
            grad_boost_alpha=grad_boost_alpha,
            grad_boost_max=grad_boost_max,
            variance_floor=variance_floor,
            conf_threshold=conf_threshold,
            conf_beta=conf_beta,
            loss_ema_beta=loss_ema_beta,
            loss_temperature=loss_temperature,
            loss_boost=loss_boost,
            gamma_plateau=gamma_plateau,
            plateau_beta=plateau_beta,
            plateau_threshold=plateau_threshold,
            recall_alpha=recall_alpha,
            hard_beta=hard_beta,
            hard_conf_threshold=hard_conf_threshold,
            max_recall_scale=max_recall_scale,
            class_ema_beta=class_ema_beta,
            class_boost_alpha=class_boost_alpha,
            class_boost_max=class_boost_max,
            confidence_gamma=confidence_gamma,
            min_conf_scale=min_conf_scale,
            evt_margin=evt_margin,
            evt_lambda_start=evt_lambda_start,
            evt_lambda_end=evt_lambda_end,
            loss_alpha=loss_alpha,
            loss_beta=loss_beta,
            kl_weight=kl_weight,
            sep_gamma=sep_gamma,
            lambda_sep=lambda_sep,
            lambda_dist=lambda_dist,
            confidence_temperature=1.5,
            late_stage_lr_boost=1.2,
            late_stage_start=0.7,
            curvature_mix=curvature_mix,
            lr_min_scale=lr_min_scale,
            lr_max_scale=lr_max_scale,
        )
        super().__init__(params, defaults)

        self._ema_cls = None
        self._ema_kl = None
        self._ema_recon = None
        self._ema_total_loss = None
        self._prev_ema_total_loss = None
        self._class_freq_ema = None

        self._loss_weight_history = []
        self._scale_history = []
        self._confidence_history = []
        self._plateau_history = []
        self._grad_variance_history = []
        self._difficulty_scale_history = []
        self._class_weight_history = []
        self._confidence_scale_history = []
        self._evt_lambda_history = []
        self._separation_loss_history = []
        self._distribution_loss_history = []
        self._dynamic_loss_weight_history = []

    @property
    def loss_weight_history(self):
        return self._loss_weight_history

    @property
    def scale_history(self):
        return self._scale_history

    @property
    def confidence_history(self):
        return self._confidence_history

    @property
    def plateau_history(self):
        return self._plateau_history

    @property
    def grad_variance_history(self):
        return self._grad_variance_history

    @property
    def recall_scale_history(self):
        return self._difficulty_scale_history

    @property
    def class_scale_history(self):
        return self._class_weight_history

    @property
    def confidence_scale_history(self):
        return self._confidence_scale_history

    @property
    def evt_lambda_history(self):
        return self._evt_lambda_history

    @property
    def separation_loss_history(self):
        return self._separation_loss_history

    @property
    def distribution_loss_history(self):
        return self._distribution_loss_history

    @property
    def dynamic_loss_weight_history(self):
        return self._dynamic_loss_weight_history

    @staticmethod
    def _clean_loss(value, default=0.0):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return float(default)
        return max(value, 0.0)

    def _update_loss_state(self, loss_cls, loss_kl, loss_rec, group):
        loss_cls = self._clean_loss(loss_cls)
        loss_kl = self._clean_loss(loss_kl)
        loss_rec = self._clean_loss(loss_rec)

        beta = group['loss_ema_beta']
        if self._ema_cls is None:
            self._ema_cls = max(loss_cls, 1e-9)
            self._ema_kl = max(loss_kl, 1e-9)
            self._ema_recon = max(loss_rec, 1e-9)
        else:
            self._ema_cls = beta * self._ema_cls + (1.0 - beta) * loss_cls
            self._ema_kl = beta * self._ema_kl + (1.0 - beta) * loss_kl
            self._ema_recon = beta * self._ema_recon + (1.0 - beta) * loss_rec

        ratios = torch.tensor([
            loss_cls / (self._ema_cls + 1e-9),
            loss_kl / (self._ema_kl + 1e-9),
            loss_rec / (self._ema_recon + 1e-9),
        ], dtype=torch.float32)

        temperature = max(float(group['loss_temperature']), 1e-6)
        weights = torch.softmax(ratios / temperature, dim=0)
        difficulty = torch.relu(ratios - 1.0)
        loss_scale = 1.0 + group['loss_boost'] * float(torch.dot(weights, difficulty))

        self._loss_weight_history.append(tuple(float(w) for w in weights))
        return loss_scale, loss_cls + loss_kl + loss_rec

    def _progress(self, epoch, total_epochs):
        if epoch is None or total_epochs is None or total_epochs <= 0:
            return 1.0
        return min(1.0, max(0.0, float(epoch) / float(total_epochs)))

    def _phase_scales(self, progress):
        if progress < 0.30:
            return 1.35, 0.90, 0.45, 0.60
        if progress < 0.70:
            return 1.10, 1.00, 1.00, 1.00
        return 1.00, 1.10, 1.85, 1.50

    def _latent_separation_loss(self, z, target, group):
        if z is None:
            return None

        spread = z.std(dim=0, unbiased=False).mean()
        distance_penalty = z.new_tensor(0.0)

        if target is not None:
            target = target.detach().long()
            classes = torch.unique(target)
            centers = []
            for cls in classes:
                mask = target == cls
                if mask.any():
                    centers.append(z[mask].mean(dim=0))
            if len(centers) >= 2:
                centers = torch.stack(centers, dim=0)
                center_dist = torch.pdist(centers, p=2)
                if center_dist.numel() > 0:
                    distance_penalty = 1.0 / (center_dist.mean() + 1e-6)

        return -spread + group['sep_gamma'] * distance_penalty

    def compute_loss(self, loss_cls, loss_kl, loss_rec, recon_error=None, z=None,
                     target=None, epoch=None, total_epochs=None):
        group = self.param_groups[0]
        progress = self._progress(epoch, total_epochs)
        cls_phase, rec_phase, sep_phase, dist_phase = self._phase_scales(progress)

        difficulty = loss_cls.detach() / (loss_cls.detach() + loss_rec.detach() + 1e-8)
        difficulty = torch.clamp(difficulty, 0.0, 1.0)

        w_cls = (1.0 + group['loss_alpha'] * float(difficulty.item())) * cls_phase
        w_rec = (1.0 + group['loss_beta'] * (1.0 - float(difficulty.item()))) * rec_phase
        w_kl = group['kl_weight']

        total_loss = w_cls * loss_cls + w_kl * loss_kl + w_rec * loss_rec

        self._dynamic_loss_weight_history.append((float(w_cls), float(w_kl), float(w_rec)))

        lambda_evt = (
            group['evt_lambda_start']
            + (group['evt_lambda_end'] - group['evt_lambda_start']) * progress
        )

        if recon_error is not None:
            spread = recon_error.detach().std(unbiased=False)
            lambda_evt = lambda_evt * (1.0 + float(spread.item()))

            mean_recon = recon_error.detach().mean()
            margin = group['evt_margin']
            evt_sep_loss = torch.relu(margin - torch.abs(recon_error - mean_recon)).mean()
            distribution_loss = recon_error.std(unbiased=False)

            total_loss = total_loss + lambda_evt * evt_sep_loss
            total_loss = total_loss + group['lambda_dist'] * dist_phase * distribution_loss

            self._evt_lambda_history.append(float(lambda_evt))
            self._distribution_loss_history.append(float(distribution_loss.detach().item()))

        latent_sep_loss = self._latent_separation_loss(z, target, group)
        if latent_sep_loss is not None:
            total_loss = total_loss + group['lambda_sep'] * sep_phase * latent_sep_loss
            self._separation_loss_history.append(float(latent_sep_loss.detach().item()))

        return total_loss

    def objective_loss(self, base_loss, recon_error, epoch=None, total_epochs=None):
        if recon_error is None:
            return base_loss

        group = self.param_groups[0]
        progress = self._progress(epoch, total_epochs)
        lambda_evt = (
            group['evt_lambda_start']
            + (group['evt_lambda_end'] - group['evt_lambda_start']) * progress
        )

        spread = recon_error.detach().std(unbiased=False)
        lambda_evt = lambda_evt * (1.0 + float(spread.item()))

        mean_recon = recon_error.detach().mean()
        margin = group['evt_margin']
        separation_loss = torch.relu(margin - torch.abs(recon_error - mean_recon)).mean()

        self._evt_lambda_history.append(float(lambda_evt))
        return base_loss + lambda_evt * separation_loss

    def _update_plateau_state(self, total_loss, group):
        total_loss = max(float(total_loss), 1e-9)
        beta = group['plateau_beta']

        if self._ema_total_loss is None:
            self._ema_total_loss = total_loss
            self._prev_ema_total_loss = total_loss
            plateau_score = 0.0
        else:
            self._prev_ema_total_loss = self._ema_total_loss
            self._ema_total_loss = beta * self._ema_total_loss + (1.0 - beta) * total_loss
            relative_delta = abs(self._prev_ema_total_loss - self._ema_total_loss)
            relative_delta /= abs(self._prev_ema_total_loss) + 1e-9
            plateau_score = 1.0 - min(relative_delta / group['plateau_threshold'], 1.0)
            plateau_score = float(max(0.0, plateau_score))

        self._plateau_history.append(plateau_score)
        return plateau_score

    def _recall_sensitive_scale(self, logits, target, group):
        if logits is None or target is None:
            return 1.0

        logits = logits.detach()
        target = target.detach().long()
        if logits.ndim != 2 or target.numel() == 0:
            return 1.0

        temperature = float(group['confidence_temperature'])
        probs = torch.softmax(logits / temperature, dim=1)
        confidence, pred = probs.max(dim=1)
        misclassified = (pred != target).float()
        hard_samples = (confidence < group['hard_conf_threshold']).float()

        boost = 1.0 + 0.6 * misclassified + 0.4 * hard_samples

        n_classes = logits.size(1)
        target = target.clamp(min=0, max=n_classes - 1)
        counts = torch.bincount(target, minlength=n_classes).float()
        batch_freq = (counts + 1.0) / (counts.sum() + n_classes).clamp_min(1.0)

        beta = group['class_ema_beta']
        if self._class_freq_ema is None or self._class_freq_ema.numel() != n_classes:
            self._class_freq_ema = batch_freq.cpu().clamp_min(1e-6)
        else:
            self._class_freq_ema.mul_(beta).add_(batch_freq.cpu(), alpha=1.0 - beta)
            self._class_freq_ema.clamp_(min=1e-6)

        class_freq = self._class_freq_ema.to(logits.device)
        inv_freq = 1.0 / (class_freq + 1e-6)
        inv_freq = inv_freq / inv_freq.mean().clamp_min(1e-6)
        class_weights = 1.0 + group['class_boost_alpha'] * torch.relu(inv_freq - 1.0)
        class_weights = torch.clamp(class_weights, min=1.0, max=group['class_boost_max'])
        sample_weight = class_weights[target]

        difficulty_scale = (boost * sample_weight).mean().item()
        difficulty_scale = min(group['max_recall_scale'], max(1.0, difficulty_scale))

        conf_mean = confidence.mean().item()
        if conf_mean > group['conf_threshold']:
            confidence_scale = 1.0 - group['confidence_gamma'] * (
                conf_mean - group['conf_threshold']
            )
            confidence_scale = max(group['min_conf_scale'], confidence_scale)
        else:
            confidence_scale = 1.0

        self._difficulty_scale_history.append(float(difficulty_scale))
        self._class_weight_history.append(float(sample_weight.mean().item()))
        self._confidence_scale_history.append(float(confidence_scale))
        return difficulty_scale * confidence_scale

    @torch.no_grad()
    def step(self, loss_cls=0.0, loss_rec=0.0, loss_kl=0.0, conf=0.5,
             logits=None, target=None, epoch=None, total_epochs=None, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        c = float(conf)
        if np.isnan(c) or np.isinf(c):
            c = 0.5
        c = min(1.0, max(0.0, c))
        self._confidence_history.append(c)

        first_group = self.param_groups[0]
        loss_scale, total_loss = self._update_loss_state(
            loss_cls, loss_kl, loss_rec, first_group
        )
        plateau_score = self._update_plateau_state(total_loss, first_group)
        has_recall_signal = logits is not None and target is not None
        recall_sensitive_scale = self._recall_sensitive_scale(logits, target, first_group)

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            b1, b2 = group['betas']
            eps_adam = group['eps']
            grad_var_beta = group['grad_var_beta']
            grad_boost_alpha = group['grad_boost_alpha']
            grad_boost_max = group['grad_boost_max']
            variance_floor = group['variance_floor']
            conf_threshold = group['conf_threshold']
            conf_beta = group['conf_beta']
            gamma_plateau = group['gamma_plateau']
            curvature_mix = group['curvature_mix']
            lr_min_scale = group['lr_min_scale']
            lr_max_scale = group['lr_max_scale']

            if has_recall_signal:
                confidence_scale = 1.0
            elif c > conf_threshold:
                confidence_scale = 1.0 - conf_beta * (c - conf_threshold)
                confidence_scale = max(group['min_conf_scale'], confidence_scale)
            else:
                confidence_scale = 1.0
            plateau_scale = 1.0 + gamma_plateau * plateau_score
            base_scale = loss_scale * confidence_scale * plateau_scale
            base_scale = float(max(lr_min_scale, min(lr_max_scale, base_scale)))

            # Late-stage learning boost to escape shallow minima
            if epoch is not None and total_epochs is not None:
                late_stage_start = float(group['late_stage_start'])
                if epoch > late_stage_start * float(total_epochs):
                    base_scale *= float(group['late_stage_lr_boost'])
                    base_scale = float(max(lr_min_scale, min(lr_max_scale, base_scale)))

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('CRAZYFOX does not support sparse gradients')

                effective_grad = grad * recall_sensitive_scale

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['grad_mean'] = torch.zeros_like(p)
                    state['grad_var'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                grad_mean = state['grad_mean']
                grad_var = state['grad_var']

                state['step'] += 1
                t = state['step']

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                grad_delta = effective_grad - grad_mean
                grad_mean.add_(grad_delta, alpha=1.0 - grad_var_beta)
                grad_var.mul_(grad_var_beta).addcmul_(
                    grad_delta, grad_delta, value=1.0 - grad_var_beta
                )

                exp_avg.mul_(b1).add_(effective_grad, alpha=1.0 - b1)
                exp_avg_sq.mul_(b2).addcmul_(effective_grad, effective_grad, value=1.0 - b2)

                bias_correction1 = 1.0 - b1 ** t
                bias_correction2 = 1.0 - b2 ** t
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                denom = v_hat.sqrt().add(eps_adam)

                var_norm = grad_var / (v_hat + eps_adam)
                var_norm = torch.clamp(var_norm, min=0.0, max=1.0)
                low_variance = torch.clamp(variance_floor - var_norm, min=0.0)
                low_variance = low_variance / max(variance_floor, 1e-9)
                grad_boost = 1.0 + grad_boost_alpha * low_variance
                grad_boost = torch.clamp(grad_boost, max=grad_boost_max)

                adam_direction = m_hat / denom
                curvature_direction = effective_grad / denom
                update_direction = (
                    (1.0 - curvature_mix) * adam_direction
                    + curvature_mix * curvature_direction
                )

                update = update_direction * grad_boost
                p.add_(update, alpha=-lr * base_scale)

                self._grad_variance_history.append(float(grad_var.mean().item()))
                self._scale_history.append(float(base_scale))

        return loss


class ABAOPlus(optim.Optimizer):
    """
    ABAO+ - Stability-Aware Adaptive Optimizer for Multi-Objective IDS Training.
    
    Unlike ABAO V1/V2 which apply a scalar weight to the full gradient, ABAO+
    decomposes training into three loss streams and assigns each an EMA-stabilized
    inverse-magnitude weight.
    
    Three loss streams:
    - L_td: Teacher-Distillation / classification loss (cross-entropy)
    - L_kl: KL Divergence loss (VAE regularization)
    - L_recon: Reconstruction loss (MSE)
    
    Weighting: w_i = 1/(EMA_i + ε), normalized so Σw_i = 1
    
    Training loop contract:
        w_td, w_kl, w_recon = optimizer.get_adaptive_weights(td, kl, recon)
        loss = w_td * L_td + w_kl * (beta * L_kl) + w_recon * L_recon
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-4, ema_beta=0.90, eps_loss=1e-8, max_norm=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        ema_beta=ema_beta, eps_loss=eps_loss, max_norm=max_norm)
        super().__init__(params, defaults)

        self._ema_td = None
        self._ema_kl = None
        self._ema_recon = None

        self._weight_history = []
        self._loss_history = []

    @property
    def weight_history(self):
        return self._weight_history

    @property
    def loss_history(self):
        return self._loss_history

    @property
    def ema_td(self):
        return self._ema_td

    @property
    def ema_kl(self):
        return self._ema_kl

    @property
    def ema_recon(self):
        return self._ema_recon

    def get_adaptive_weights(self, loss_td: float, loss_kl: float, loss_recon: float):
        """
        Compute adaptive weights for three loss streams.
        
        w_i = 1/(EMA_i + ε), normalized so Σw_i = 1
        
        Returns:
            (w_td, w_kl, w_recon): Normalized adaptive weights
        """
        ema_beta = self.defaults['ema_beta']
        eps_loss = self.defaults['eps_loss']

        # Moving Average Stabilization
        if self._ema_td is None:
            self._ema_td = max(float(loss_td), 1e-9)
            self._ema_kl = max(float(loss_kl), 1e-9)
            self._ema_recon = max(float(loss_recon), 1e-9)
        else:
            self._ema_td = ema_beta * self._ema_td + (1.0 - ema_beta) * float(loss_td)
            self._ema_kl = ema_beta * self._ema_kl + (1.0 - ema_beta) * float(loss_kl)
            self._ema_recon = ema_beta * self._ema_recon + (1.0 - ema_beta) * float(loss_recon)

        # Inverse-Magnitude Weighting
        w_td = 1.0 / (self._ema_td + eps_loss)
        w_kl = 1.0 / (self._ema_kl + eps_loss)
        w_recon = 1.0 / (self._ema_recon + eps_loss)

        # Normalize so they sum to 1
        total_w = w_td + w_kl + w_recon
        w_td /= total_w
        w_kl /= total_w
        w_recon /= total_w

        # Logging
        self._weight_history.append((w_td, w_kl, w_recon))
        self._loss_history.append((float(loss_td), float(loss_kl), float(loss_recon)))

        return w_td, w_kl, w_recon

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            b1, b2 = group['betas']
            eps_adam = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # Decoupled weight decay (AdamW-style)
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                # Adam moment updates
                exp_avg.mul_(b1).add_(grad, alpha=1.0 - b1)
                exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)

                # Bias-corrected estimates
                m_hat = exp_avg / (1.0 - b1 ** t)
                v_hat = exp_avg_sq / (1.0 - b2 ** t)

                # Parameter update
                denom = v_hat.sqrt().add_(eps_adam)
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss
