# si-renyi-entropy

> **Proof of Concept:** The Rényi entropy spectrum H(α) characterizes fleet diversity more richly than any single metric — detecting monocultures, measuring specialist/generalist balance, and predicting fleet fragility.

## The Insight

A single diversity number (Shannon entropy) tells you *how diverse* the fleet is. But the full **Rényi spectrum** H(α) for α ∈ [0, ∞] tells you *what kind* of diversity you have:

| α | Entropy | What It Measures |
|---|---------|-----------------|
| 0 | Hartley | How many capabilities are *present at all* |
| 1 | Shannon | Standard diversity (evenness) |
| 2 | Collision | Inverse Simpson — probability two random picks differ |
| ∞ | Min-entropy | How concentrated the *dominant* capability is |

**Flat spectrum** = uniform agents (maximum diversity). **Steep spectrum** = specialists with monoculture risk.

The **slope** of the Rényi spectrum is a single number that captures the *shape* of diversity:

- Slope ≈ 0: Healthy, balanced fleet
- Slope << 0: Monoculture — one capability dominates
- Slope varies: Mixed specialists and generalists

## What This Proves

1. **Monoculture detection**: A fleet where all agents have the same specialization scores high on monoculture risk
2. **Health score**: Shannon entropy normalized by max entropy gives a 0-1 fleet health metric
3. **Divergence measurement**: Rényi divergence between agent profiles quantifies how different agents are from each other
4. **Spectrum slope predicts fragility**: Steep negative slope = fleet fragile to capability loss

## Usage

```rust
use si_renyi_entropy::*;

// Single agent analysis
let agent = AgentProfile::new(0, vec![0.5, 0.3, 0.1, 0.1]);
println!("Shannon entropy: {}", agent.entropy());
println!("Full spectrum: {:?}", agent.spectrum());
println!("Is specialist: {}", agent.is_specialist());

// Fleet diversity
let fleet = FleetDiversity::new(vec![
    AgentProfile::new(0, vec![1.0, 0.0, 0.0]),  // specialist
    AgentProfile::new(1, vec![0.0, 1.0, 0.0]),  // specialist
    AgentProfile::new(2, vec![1.0, 1.0, 1.0]),  // generalist
]);

println!("Fleet health: {}", fleet.health_score());        // 0-1
println!("Monoculture risk: {}", fleet.monoculture_risk()); // 0-1
println!("Specialist fraction: {}", fleet.specialist_fraction());
println!("Avg divergence: {}", fleet.average_divergence(2.0));

let spectrum = fleet.fleet_spectrum();
println!("Spectrum slope: {}", spectrum_slope(&spectrum));
```

## Modules

- `renyi_entropy()` — Core Rényi entropy computation for any α
- `shannon_entropy()` — Shannon entropy (α→1 limit)
- `min_entropy()` — Min-entropy (α→∞ limit)
- `collision_entropy()` — Collision entropy (α=2)
- `simpson_index()` — Simpson diversity index
- `AgentProfile` — Single agent's capability distribution with entropy/spectrum methods
- `FleetDiversity` — Fleet-level diversity analysis (health, monoculture risk, divergence)
- `renyi_divergence()` — Rényi divergence D_α(P || Q) between distributions
- `kl_divergence()` — KL divergence (α→1 limit)
- `jensen_renyi_divergence()` — Symmetric Jensen-Rényi divergence
- `spectrum_slope()` — Linear regression slope of H(α) curve

## Connection to Conservation Law

The conservation law γ + η = C constrains *total* budget. Rényi entropy measures whether that budget is *well-distributed*:

- **High H(0)**: Budget spread across many capabilities (robust)
- **Low H(∞)**: One capability dominates (fragile)
- **Spectrum slope**: How quickly dominance emerges as you focus on extremes

Conservation without diversity = monoculture. Diversity without conservation = waste. You need both.

## Mathematical Background

### Rényi Entropy
For a discrete distribution P = (p₁, ..., pₙ):

H_α(P) = 1/(1-α) · ln(Σ pᵢᵅ)

Special cases:
- α→0: H₀ = ln(|{i : pᵢ > 0}|) — support size
- α→1: H₁ = -Σ pᵢ ln(pᵢ) — Shannon entropy
- α→2: H₂ = -ln(Σ pᵢ²) — collision entropy
- α→∞: H∞ = -ln(max pᵢ) — min-entropy

**Key property**: H(α) is non-increasing in α. The *shape* of the curve carries more information than any point.

### Rényi Divergence
D_α(P || Q) = 1/(α-1) · ln(Σ pᵢᵅ qᵢ¹⁻ᵅ)

Measures how different P is from Q. Not symmetric (use Jensen-Rényi for symmetry).

### Fleet Health Score
health = H₁(fleet_dist) / ln(n_capabilities)

Normalized Shannon entropy. 1.0 = perfectly uniform, 0.0 = total monoculture.

## Tests: 21

Covers: uniform max entropy, specialist low entropy, Rényi monotonicity, all α values, agent profiles, fleet distribution, health scores, monoculture risk, divergence rankings, KL self-divergence, Jensen-Rényi symmetry, spectrum slopes.

## License

MIT
