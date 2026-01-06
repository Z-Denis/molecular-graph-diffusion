"""Quick checks for reverse diffusion stability on a trained model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from mgd.dataset import GraphBatchLoader
from mgd.diffusion.schedules import DiffusionSchedule
from mgd.latent import GraphLatent
from mgd.sampling.updater import DDPMUpdater
from mgd.training import DiffusionTrainState


def masked_mean_std(latent: GraphLatent, node_mask, pair_mask, eps=1e-8):
    nw = jnp.broadcast_to(node_mask[..., None], latent.node.shape)
    ew = jnp.broadcast_to(pair_mask[..., None], latent.edge.shape)

    def stats(x, w):
        mean = (x * w).sum() / jnp.maximum(w.sum(), 1.0)
        var = ((x - mean) ** 2 * w).sum() / jnp.maximum(w.sum(), 1.0)
        return mean, jnp.sqrt(var + eps)

    return stats(latent.node, nw), stats(latent.edge, ew)


def one_step_reverse_sigma0(
    state: DiffusionTrainState,
    batch,
    t: int,
    rng: jax.Array,
):
    """Single reverse step with trained eps_hat and sigma=0 to check scale."""
    x0 = state.model.apply({"params": state.params}, batch, method=state.model.encode)
    rng, kn, ke = jax.random.split(rng, 3)
    noise = GraphLatent(
        jax.random.normal(kn, x0.node.shape),
        jax.random.normal(ke, x0.edge.shape),
    )
    t_arr = jnp.full((batch.atom_type.shape[0],), t, dtype=jnp.int32)
    xt = state.model.q_sample(x0, t_arr, noise)

    eps_hat = state.model.apply(
        {"params": state.params},
        xt,
        t_arr,
        node_mask=batch.node_mask,
        pair_mask=batch.pair_mask,
        method=state.model.predict_eps,
    )

    beta_t = jnp.take(state.model.schedule.betas, t)
    alpha_t = jnp.take(state.model.schedule.alphas, t)
    alpha_bar_t = jnp.take(state.model.schedule.alpha_bar, t)
    coef1 = 1.0 / jnp.sqrt(alpha_t)
    coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)

    x_prev = GraphLatent(
        coef1 * (xt.node - coef2 * eps_hat.node),
        coef1 * (xt.edge - coef2 * eps_hat.edge),
    ).masked(batch.node_mask, batch.pair_mask)

    stats_noise = masked_mean_std(noise, batch.node_mask, batch.pair_mask)
    stats_eps = masked_mean_std(eps_hat, batch.node_mask, batch.pair_mask)
    stats_prev = masked_mean_std(x_prev, batch.node_mask, batch.pair_mask)
    return {
        "noise_stats": stats_noise,
        "eps_hat_stats": stats_eps,
        "x_prev_stats": stats_prev,
    }


def deterministic_reverse(
    state: DiffusionTrainState,
    batch,
    n_steps: int,
    rng: jax.Array,
):
    """Run a deterministic reverse (sigma=0) to check stability."""
    updater = DDPMUpdater(state.model.schedule)

    def predict_eps(xt, t, nm, pm):
        return state.model.apply(
            {"params": state.params},
            xt,
            t,
            node_mask=nm,
            pair_mask=pm,
            method=state.model.predict_eps,
        )

    xt = state.model.apply({"params": state.params}, batch, method=state.model.encode)
    times = jnp.arange(n_steps, 0, -1, dtype=jnp.int32)

    def body(carry, t):
        xt_c = carry
        eps_hat = predict_eps(xt_c, jnp.full(xt_c.node.shape[:-2], t, dtype=jnp.int32), batch.node_mask, batch.pair_mask)
        beta_t = jnp.take(state.model.schedule.betas, t)
        alpha_t = jnp.take(state.model.schedule.alphas, t)
        alpha_bar_t = jnp.take(state.model.schedule.alpha_bar, t)
        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
        xt_next = GraphLatent(
            coef1 * (xt_c.node - coef2 * eps_hat.node),
            coef1 * (xt_c.edge - coef2 * eps_hat.edge),
        ).masked(batch.node_mask, batch.pair_mask)
        return xt_next, xt_next

    xt_final, traj = jax.lax.scan(body, xt, times)
    return xt_final, traj


if __name__ == "__main__":
    # Example usage (replace loader/state restore with your own):
    # state, batch = load_trained_state()
    # rng = jax.random.PRNGKey(0)
    # print(one_step_reverse_sigma0(state, batch, t=500, rng=rng))
    pass
