# Comparison of Forces with and without $\bar{V}$ and Node Warp

## Hellmann-Feynman Force

So far, we have observed that the node warp transformation succesfully eliminates divergences
in the Hellman-Feynman force,

\begin{align}
	F_{HF} = \langle \nabla_a E_L(x) \rangle,
\end{align}

where $a$ denotes the geometric parameters of the system.
This is exactly how the node warp transformation is constructed. If $d(x)$ denotes the
distance to the node from the configuration $x$, the local energy near the node
behaves as

\begin{align}
	E_L \sim \frac{1}{d(x)}.
\end{align}

If the geometric parameter $a$ changes the nodal surface, we have

\begin{align}
	\partial_a E_L \sim -\frac{1}{d^2}\frac{\partial d}{\partial a},
\end{align}

which has an infinite variance. The node warp transformation is designed to eliminate the
dependence of $d$ on $a$, causing the derivative to vanish, leaving $\partial a E_L$ finite.

Let $d'(x)$ denote the distance of $x$ to the nodes of a secondary geometry with
$a \to a + \Delta a$. Then, in warped coordinates, we have

\begin{align}
	\frac{\partial d(\bar{x})}{\partial a} = \lim_{\Delta a \to 0} 
		\frac{d'(\bar{x}) - d(\bar{x})}{\Delta a}.
\end{align}

By construcion, $d'(\bar{x}) = d(x)$. As $\Delta a \to 0$, we have $\bar{x} \to x$, and so

\begin{align}
	\frac{\partial d(\bar{x})}{\partial a} = \lim_{\Delta a \to 0} 
		\frac{d'(\bar{x}) - d(x)}{\Delta a} = 0.
\end{align}

This clearly helps reduce the variance of the Hellmann-Feynman force term.

## Pulay Force

The situation is more complicated with the Pulay force. In the VD approximation, the Pulay force
is given by

\begin{align}
	F_P = \left\langle (E_L - E)
		\left[2\nabla_a \ln \Psi + \sum_{i=n-k}^{n-1} \nabla_a S(x_{i+1}, x_i)\right]
	\right\rangle.
\end{align}

Focusing on the first term in the brackets, we have

\begin{align}
	\nabla_a \ln \Psi = \frac{\nabla_a\Psi}{\Psi} \sim \frac{1}{d(x)}.
\end{align}

Hence, this term behaves similarly to the local energy. We thus have the same divergence-cancelling
effect of the node warp transformation. This explains why the Pulay force behaves quite well for
the VD approximation.

![The Pulay and Hellmann-Feynman forces for the VD approximation in warped coordinates, with and without the relocity cutoff.](VD_warp.eps){ width=600px }

Using the exact formula, we instead find a very noisy result, with the warp transformation
actually making the fluctuations worse. As far as I can see this is because the Pulay
force in the exact formulation *never contains the drift velocity*. To see this, consider the
noisy term,

\begin{align}
	\nabla_a T(x_{i+1}, x_i),
\end{align}

where $x_{i+1}$ and $x_i$ denote successive configurations, and 
$\exp(T)$ is the drift-diffusion part of the short-time Green's function. If the move from 
$x_i$ to $x_{i+1}$ was rejected, we set T = 0$. Otherwise, we have

\begin{align}
	x_{i+1} = x_i + \bar{V}(x_i)\tau + \sqrt{\tau}\Delta W,
\end{align}

with $\Delta W$ a Gaussian random vector with unit variance and zero mean. Thus,

\begin{align}
	x_{i+1} - x_i - \bar{V}(x_i)\tau = \sqrt{\tau}\Delta W.
\end{align}

If we do not use the node warp, since both the primary and secondary geometry use the same random
walk, we end up with $\nabla_a T = 0$. Otherwise, using the node warp, we *no longer have this
cancellation*, since $\bar{x}$ is a nonlinear function of $x$. It is not clear to
me how to correct $T$ for this effect.

## Cutoff Velocity

Since the velocity $V(x)$ also diverges near the nodes, it is  illustrative to examine
the effect that the velocity cutoff has on the fluctuations of the Hellmann-Feynman and Pulay
force terms, with and without the node warp.

![Exact forces without node warp](exact_nowarp.eps){ width=600px }

![Exact forces with node warp](exact_warp.eps){ width=600px }

![VD forces without node warp](VD_nowarp.eps){ width=600px }

![VD forces with node warp](VD_nowarp.eps){ width=600px }


It is clear from these figures that the cutoff velocity does indeed have a regularizing effect
on the forces, especially in the case of the exact Pulay force without a node warp in place.

## HF Force and Cutoff $\bar{V}$ Combined

Especially striking is the improvement observed when using both the cutoff velocity and
the node warp transformation in the Hellmann-Feynman force:

![Hellmann-Feynman force with both cutoff velocity and node warp.](HF_cutoff_warp.eps){ width=600px }
