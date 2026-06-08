//! Rényi entropy spectrum for measuring agent diversity and detecting monocultures.
//!
//! H_α = 1/(1-α) · log(Σ p_i^α)
//! - α→0: Hartley entropy (count of nonzero skills)
//! - α→1: Shannon entropy (diversity)
//! - α→2: Collision entropy (inverse Simpson)
//! - α→∞: Min-entropy (most concentrated skill)

pub fn renyi_entropy(probs: &[f64], alpha: f64) -> f64 {
    if alpha.abs() < 1e-10 {
        let count = probs.iter().filter(|p| **p > 1e-12).count() as f64;
        if count > 0.0 { count.ln() } else { 0.0 }
    } else if (alpha - 1.0).abs() < 1e-10 {
        shannon_entropy(probs)
    } else if alpha.is_infinite() && alpha.is_sign_positive() {
        min_entropy(probs)
    } else {
        let sum: f64 = probs.iter().filter(|p| **p > 1e-12).map(|p| p.powf(alpha)).sum();
        if sum < 1e-12 { 0.0 } else { (1.0 / (1.0 - alpha)) * sum.ln() }
    }
}

pub fn shannon_entropy(probs: &[f64]) -> f64 {
    probs.iter().filter(|p| **p > 1e-12).map(|p| -p * p.ln()).sum()
}

pub fn renyi_spectrum(probs: &[f64], alphas: &[f64]) -> Vec<(f64, f64)> {
    alphas.iter().map(|&a| (a, renyi_entropy(probs, a))).collect()
}

pub fn standard_alphas() -> Vec<f64> {
    vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, f64::INFINITY]
}

pub fn min_entropy(probs: &[f64]) -> f64 {
    let max_p = probs.iter().cloned().fold(0.0_f64, f64::max);
    if max_p < 1e-12 { 0.0 } else { -max_p.ln() }
}

pub fn collision_entropy(probs: &[f64]) -> f64 { renyi_entropy(probs, 2.0) }

pub fn simpson_index(probs: &[f64]) -> f64 { 1.0 - probs.iter().map(|p| p * p).sum::<f64>() }

#[derive(Debug, Clone)]
pub struct AgentProfile { pub id: usize, pub capabilities: Vec<f64> }

impl AgentProfile {
    pub fn new(id: usize, capabilities: Vec<f64>) -> Self { Self { id, capabilities } }
    pub fn normalized(&self) -> Vec<f64> {
        let sum: f64 = self.capabilities.iter().sum();
        if sum < 1e-12 { vec![0.0; self.capabilities.len()] }
        else { self.capabilities.iter().map(|c| c / sum).collect() }
    }
    pub fn entropy(&self) -> f64 { shannon_entropy(&self.normalized()) }
    pub fn renyi(&self, alpha: f64) -> f64 { renyi_entropy(&self.normalized(), alpha) }
    pub fn spectrum(&self) -> Vec<(f64, f64)> { renyi_spectrum(&self.normalized(), &standard_alphas()) }
    pub fn effective_breadth(&self) -> usize { self.capabilities.iter().filter(|c| **c > 1e-6).count() }
    pub fn is_specialist(&self) -> bool { self.effective_breadth() <= 3 }
}

#[derive(Debug, Clone)]
pub struct FleetDiversity { pub agents: Vec<AgentProfile> }

impl FleetDiversity {
    pub fn new(agents: Vec<AgentProfile>) -> Self { Self { agents } }
    pub fn fleet_distribution(&self) -> Vec<f64> {
        let n = self.agents.len() as f64;
        let dim = self.agents[0].capabilities.len();
        let mut avg = vec![0.0; dim];
        for agent in &self.agents {
            let norm = agent.normalized();
            for (i, w) in norm.iter().enumerate() { avg[i] += w / n; }
        }
        avg
    }
    pub fn fleet_entropy(&self, alpha: f64) -> f64 { renyi_entropy(&self.fleet_distribution(), alpha) }
    pub fn fleet_spectrum(&self) -> Vec<(f64, f64)> { renyi_spectrum(&self.fleet_distribution(), &standard_alphas()) }
    pub fn average_divergence(&self, alpha: f64) -> f64 {
        let n = self.agents.len();
        if n < 2 { return 0.0; }
        let mut total = 0.0; let mut count = 0;
        for i in 0..n { for j in (i+1)..n {
            total += renyi_divergence(&self.agents[i].normalized(), &self.agents[j].normalized(), alpha);
            count += 1;
        }}
        total / count as f64
    }
    pub fn monoculture_risk(&self) -> f64 {
        let d = self.fleet_distribution();
        let h0 = renyi_entropy(&d, 0.0);
        let hinf = min_entropy(&d);
        if h0 < 1e-12 { 1.0 } else { 1.0 - hinf / h0 }
    }
    pub fn specialist_fraction(&self) -> f64 {
        self.agents.iter().filter(|a| a.is_specialist()).count() as f64 / self.agents.len() as f64
    }
    pub fn health_score(&self) -> f64 {
        let d = self.fleet_distribution();
        let h = shannon_entropy(&d);
        let hmax = (d.len() as f64).ln();
        if hmax < 1e-12 { 0.0 } else { h / hmax }
    }
}

pub fn renyi_divergence(p: &[f64], q: &[f64], alpha: f64) -> f64 {
    if alpha.abs() < 1e-10 {
        let sum: f64 = p.iter().zip(q.iter()).map(|(pi,qi)| pi.min(*qi)).sum();
        if sum < 1e-12 { f64::INFINITY } else { -sum.ln() }
    } else if (alpha - 1.0).abs() < 1e-10 {
        kl_divergence(p, q)
    } else {
        let sum: f64 = p.iter().zip(q.iter())
            .filter(|(pi,qi)| **pi > 1e-12 && **qi > 1e-12)
            .map(|(pi,qi)| pi.powf(alpha) * qi.powf(1.0-alpha)).sum();
        if sum < 1e-12 { f64::INFINITY } else { (1.0/(alpha-1.0)) * sum.ln() }
    }
}

pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter())
        .filter(|(pi,qi)| **pi > 1e-12 && **qi > 1e-12)
        .map(|(pi,qi)| pi * (pi/qi).ln()).sum()
}

pub fn jensen_renyi_divergence(p: &[f64], q: &[f64], alpha: f64) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(a,b)| (a+b)/2.0).collect();
    renyi_entropy(&m, alpha) - (renyi_entropy(p, alpha) + renyi_entropy(q, alpha)) / 2.0
}

pub fn spectrum_slope(spectrum: &[(f64, f64)]) -> f64 {
    let pts: Vec<(f64,f64)> = spectrum.iter().filter(|(a,_)| a.is_finite()).cloned().collect();
    if pts.len() < 2 { return 0.0; }
    let n = pts.len() as f64;
    let sx: f64 = pts.iter().map(|(x,_)| *x).sum();
    let sy: f64 = pts.iter().map(|(_,y)| *y).sum();
    let sxy: f64 = pts.iter().map(|(x,y)| x*y).sum();
    let sx2: f64 = pts.iter().map(|(x,_)| x*x).sum();
    let d = n*sx2 - sx*sx;
    if d.abs() < 1e-12 { 0.0 } else { (n*sxy - sx*sy) / d }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn uniform(n: usize) -> Vec<f64> { vec![1.0/n as f64; n] }
    fn spec(n: usize, f: usize) -> Vec<f64> {
        let mut p = vec![0.01; n]; p[f % n] = 1.0;
        let s: f64 = p.iter().sum(); p.iter().map(|x| x/s).collect()
    }

    #[test] fn uniform_max_entropy() { assert!((shannon_entropy(&uniform(4)) - 4.0_f64.ln()).abs() < 1e-10); }
    #[test] fn specialist_low_entropy() { assert!(shannon_entropy(&spec(4,0)) < 0.5); }
    #[test] fn renyi_decreasing() {
        let s = renyi_spectrum(&[0.3,0.3,0.2,0.2], &[0.0,0.5,1.0,2.0,5.0,10.0]);
        for i in 1..s.len() { assert!(s[i].1 <= s[i-1].1 + 0.01); }
    }
    #[test] fn renyi_alpha_0() { assert!((renyi_entropy(&[0.0,0.5,0.0,0.5], 0.0) - 2.0_f64.ln()).abs() < 1e-10); }
    #[test] fn min_entropy_test() { assert!((min_entropy(&[0.1,0.7,0.1,0.1]) - (-0.7_f64.ln())).abs() < 1e-10); }
    #[test] fn collision_uniform() { assert!((collision_entropy(&uniform(4)) - 4.0_f64.ln()).abs() < 0.1); }
    #[test] fn simpson_uniform() { assert!((simpson_index(&uniform(4)) - 0.75).abs() < 1e-10); }
    #[test] fn agent_entropy_ranking() {
        assert!(AgentProfile::new(0,vec![1.0,1.0,1.0,1.0]).entropy() > AgentProfile::new(1,vec![10.0,0.1,0.1,0.1]).entropy());
    }
    #[test] fn agent_specialist() {
        assert!(AgentProfile::new(0,vec![10.0,0.001,0.001]).is_specialist());
        assert!(!AgentProfile::new(1,vec![1.0,1.0,1.0,1.0,1.0]).is_specialist());
    }
    #[test] fn fleet_dist() {
        let d = FleetDiversity::new(vec![AgentProfile::new(0,vec![1.0,0.0]),AgentProfile::new(1,vec![0.0,1.0])]).fleet_distribution();
        assert!((d[0]-0.5).abs()<1e-10 && (d[1]-0.5).abs()<1e-10);
    }
    #[test] fn health_uniform() {
        let f = FleetDiversity::new(vec![AgentProfile::new(0,vec![1.0,1.0,1.0,1.0]),AgentProfile::new(1,vec![1.0,1.0,1.0,1.0])]);
        assert!((f.health_score()-1.0).abs()<1e-10);
    }
    #[test] fn health_monoculture() {
        let f = FleetDiversity::new(vec![AgentProfile::new(0,vec![1.0,0.0,0.0,0.0]),AgentProfile::new(1,vec![1.0,0.0,0.0,0.0])]);
        assert!(f.health_score()<0.1);
    }
    #[test] fn monoculture_risk_ranking() {
        let div = FleetDiversity::new((0..5).map(|i| AgentProfile::new(i,spec(5,i))).collect());
        let mono = FleetDiversity::new(vec![AgentProfile::new(0,vec![1.0,0.0,0.0]);3]);
        assert!(div.monoculture_risk() < mono.monoculture_risk());
    }
    #[test] fn divergence_ranking() {
        let d = FleetDiversity::new(vec![AgentProfile::new(0,vec![1.0,0.0,0.0]),AgentProfile::new(1,vec![0.0,1.0,0.0])]);
        let s = FleetDiversity::new(vec![AgentProfile::new(0,vec![1.0,0.0,0.0]),AgentProfile::new(1,vec![1.0,0.0,0.0])]);
        assert!(d.average_divergence(2.0) > s.average_divergence(2.0));
    }
    #[test] fn kl_self_zero() { assert!(kl_divergence(&[0.5,0.5],&[0.5,0.5]) < 1e-10); }
    #[test] fn renyi_div_positive() { assert!(renyi_divergence(&[1.0,0.0],&[0.5,0.5],2.0) > 0.0); }
    #[test] fn jensen_symmetric() {
        let d1=jensen_renyi_divergence(&[0.9,0.1],&[0.1,0.9],2.0);
        let d2=jensen_renyi_divergence(&[0.1,0.9],&[0.9,0.1],2.0);
        assert!((d1-d2).abs()<1e-10);
    }
    #[test] fn slope_flat() { assert!(spectrum_slope(&renyi_spectrum(&uniform(8),&[0.0,1.0,2.0,5.0,10.0])).abs()<0.1); }
    #[test] fn slope_concentrated() { assert!(spectrum_slope(&renyi_spectrum(&spec(8,0),&[0.0,1.0,2.0,5.0,10.0])) < -0.1); }
    #[test] fn specialist_frac() {
        let f = FleetDiversity::new(vec![
            AgentProfile::new(0,vec![10.0,0.01,0.01]),AgentProfile::new(1,vec![1.0,1.0,1.0,1.0]),AgentProfile::new(2,vec![0.01,10.0,0.01])]);
        assert!((f.specialist_fraction()-2.0/3.0).abs()<0.01);
    }
    #[test] fn five_diverse() {
        let f = FleetDiversity::new((0..5).map(|i| AgentProfile::new(i,spec(5,i))).collect());
        assert!(f.health_score() > 0.5);
        assert!(f.monoculture_risk() < 0.5);
    }
}
