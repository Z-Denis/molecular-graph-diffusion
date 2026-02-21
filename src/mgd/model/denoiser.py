"""Flax-based molecular graph diffusion denoiser."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..dataset.chemistry import ChemistrySpec, DEFAULT_CHEMISTRY
from ..latent import GraphLatent, symmetrize_edge
from .backbone import MPNNBackbone, TransformerBackbone


class MPNNDenoiser(nn.Module):
    node_dim: int
    edge_dim: int
    
    mess_dim: int   # message hidden dim
    time_dim: int   # time hidden dim
    node_count_dim: int | None = None

    spec: ChemistrySpec = DEFAULT_CHEMISTRY

    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    n_layers: int = 1
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(
        self,
        xt: GraphLatent,
        log_sigma: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> GraphLatent:
        backbone = MPNNBackbone(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            mess_dim=self.mess_dim,
            time_dim=self.time_dim,
            node_count_dim=self.node_count_dim,
            max_nodes=self.spec.max_nodes,
            activation=self.activation,
            param_dtype=self.param_dtype,
            n_layers=self.n_layers,
            name="backbone",
        )

        # Evaluate heads
        nodes, edges = backbone(
            xt.node,
            xt.edge,
            log_sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )

        # Latent to noise space
        eps_nodes = nn.Dense(
            self.spec.atom_vocab_size, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="node_head",
        )(nodes)
        eps_edges = nn.Dense(
            self.spec.bond_vocab_size, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="edge_head",
        )(edges)

        # Mask out invalid atomic positions
        node_mask = jax.lax.stop_gradient(node_mask)
        pair_mask = jax.lax.stop_gradient(pair_mask)
        eps_nodes = eps_nodes * node_mask[..., None]
        eps_edges = eps_edges * pair_mask[..., None]

        return GraphLatent(eps_nodes, eps_edges)


class TransformerDenoiser(nn.Module):
    node_dim: int
    edge_dim: int

    time_dim: int   # time hidden dim
    node_count_dim: int | None = None

    spec: ChemistrySpec = DEFAULT_CHEMISTRY

    n_layers: int = 1
    n_heads: int = 8
    alpha_node: int = 4
    alpha_edge: int = 4
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    param_dtype: DTypeLike = "float32"
    symmetrize: bool = True
    use_mul: bool = True

    @nn.compact
    def __call__(
        self,
        xt: GraphLatent,
        log_sigma: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> GraphLatent:
        backbone = TransformerBackbone(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            time_dim=self.time_dim,
            node_count_dim=self.node_count_dim,
            max_nodes=self.spec.max_nodes,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            alpha_node=self.alpha_node,
            alpha_edge=self.alpha_edge,
            activation=self.activation,
            param_dtype=self.param_dtype,
            symmetrize=self.symmetrize,
            use_mul=self.use_mul,
            name="backbone",
        )

        # Evaluate heads
        nodes, edges = backbone(
            xt.node,
            xt.edge,
            log_sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )

        # Latent to noise space
        eps_nodes = nn.Dense(
            self.spec.atom_vocab_size, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="node_head",
        )(nodes)
        eps_edges = nn.Dense(
            self.spec.bond_vocab_size, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="edge_head",
        )(edges)

        # Mask out invalid atomic positions
        node_mask = jax.lax.stop_gradient(node_mask)
        pair_mask = jax.lax.stop_gradient(pair_mask)
        eps_nodes = eps_nodes * node_mask[..., None]
        eps_edges = eps_edges * pair_mask[..., None]

        return GraphLatent(eps_nodes, eps_edges)


class GatedMPNNDenoiser(nn.Module):
    node_dim: int
    edge_dim: int

    mess_dim: int
    time_dim: int
    node_count_dim: int | None = None

    spec: ChemistrySpec = DEFAULT_CHEMISTRY

    beta_min: float = 0.0
    beta_max: float = 8.0
    beta_k: float = 3.0
    sigma_data: float = 1.0
    gate_eps: float = 1e-8
    use_routing_gate: bool = False
    routing_beta_min: float = 0.0
    routing_beta_max: float = 1.0
    routing_beta_k: float = 3.0
    routing_sigma_data: float = 1.0

    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    n_layers: int = 1
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(
        self,
        xt: GraphLatent,
        log_sigma: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> GraphLatent:
        backbone = MPNNBackbone(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            mess_dim=self.mess_dim,
            time_dim=self.time_dim,
            node_count_dim=self.node_count_dim,
            max_nodes=self.spec.max_nodes,
            use_routing_gate=self.use_routing_gate,
            routing_beta_min=self.routing_beta_min,
            routing_beta_max=self.routing_beta_max,
            routing_beta_k=self.routing_beta_k,
            routing_sigma_data=self.routing_sigma_data,
            activation=self.activation,
            param_dtype=self.param_dtype,
            n_layers=self.n_layers,
            name="backbone",
        )

        nodes, edges = backbone(
            xt.node,
            xt.edge,
            log_sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )

        node_logits = nn.Dense(
            self.spec.atom_vocab_size,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="node_head",
        )(nodes)
        edge_logits = nn.Dense(
            self.spec.bond_vocab_size,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="edge_head",
        )(edges)

        node_probs = jax.nn.softmax(node_logits, axis=-1)
        capacity_table = jnp.asarray(self.spec.valence_table, dtype=node_logits.dtype)
        capacity = jnp.einsum("bnc,c->bn", node_probs, capacity_table) * node_mask

        exist_logits_sym = 0.5 * (
            edge_logits[..., 0] + jnp.swapaxes(edge_logits[..., 0], -1, -2)
        )
        type_logits_sym = symmetrize_edge(edge_logits[..., 1:])
        p_exist = jax.nn.sigmoid(exist_logits_sym)
        p_type = jax.nn.softmax(type_logits_sym, axis=-1)

        bond_orders = jnp.asarray(self.spec.bond_orders, dtype=edge_logits.dtype)
        n_type = edge_logits.shape[-1] - 1
        if bond_orders.shape[0] == n_type + 1:
            bond_orders = bond_orders[1:]
        elif bond_orders.shape[0] != n_type:
            raise ValueError(
                "spec.bond_orders must match edge type channels "
                f"(got {bond_orders.shape[0]}, expected {n_type} or {n_type + 1})."
            )
        bond_mass = pair_mask * p_exist * jnp.einsum("bnmk,k->bnm", p_type, bond_orders)

        used = jnp.sum(bond_mass, axis=-1)
        residual = capacity - used

        log_sigma_data = jnp.log(jnp.asarray(self.sigma_data, dtype=edge_logits.dtype))
        sharp = jax.nn.sigmoid(self.beta_k * (log_sigma_data - log_sigma))
        beta = self.beta_min + (self.beta_max - self.beta_min) * sharp

        g_node = jax.nn.sigmoid(beta[..., None] * residual) * node_mask
        g_edge = pair_mask * (g_node[..., :, None] * g_node[..., None, :])
        exist_bias = jnp.log(jnp.maximum(g_edge, self.gate_eps))

        edge_logits = edge_logits.at[..., 0].add(exist_bias)

        node_mask = jax.lax.stop_gradient(node_mask)
        pair_mask = jax.lax.stop_gradient(pair_mask)
        node_logits = node_logits * node_mask[..., None]
        edge_logits = edge_logits * pair_mask[..., None]

        return GraphLatent(node_logits, edge_logits)


__all__ = ["MPNNDenoiser", "TransformerDenoiser", "GatedMPNNDenoiser"]
