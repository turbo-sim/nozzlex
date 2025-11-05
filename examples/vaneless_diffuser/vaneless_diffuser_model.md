## Original model

The original model describes steady, one-dimensional flow along the mean streamline of an annular channel. The governing equations are:


$$
\begin{gather}
v_m \frac{d\rho}{dm} + \rho \frac{dv_m}{dm}
  = -\frac{\rho v_m}{A} \frac{dA}{dm} \\[4pt]
\rho v_m \frac{dv_m}{dm} + \frac{dp}{dm}
  = \frac{\rho v_\theta^2}{r} \sin(\phi)
     - \frac{2 \tau}{b} \cos(\alpha) \\[4pt]
\rho v_m \frac{dv_\theta}{dm}
  = -\frac{\rho v_\theta v_m}{r} \sin(\phi)
     - \frac{2 \tau}{b} \sin(\alpha) \\[4pt]
\rho v_m \frac{dp}{dm} - \rho v_m a^2 \frac{d\rho}{dm}
  = \frac{2(\tau v + \dot{q})}{b \left( \frac{\partial e}{\partial p} \right)_\rho}
\end{gather}
$$

The first equation expresses the **mass balance** in the annular passage. The second and third equations represent the **momentum balance** in the meridional and tangential directions, respectively. The last equation represents **thermal energy balance**, incorporating viscous dissipation and wall heat transfer. The viscous shear stress is modeled using a conventional friction factor approach:
$$
\tau = \frac{1}{2} c_f \rho v^2
$$
where $c_f$ is the skin friction coefficient, obtained from empirical correlations as a function of the Reynolds number. Thermodynamic properties are evaluated from the **equation of state**, using pressure–density as input pair.

> "R. Agromayor, B. Müller, and L. O. Nord, “One-dimensional annular diffuser model for preliminary turbomachinery design,” International Journal of Turbomachinery, Propulsion and Power, vol. 4, no. 3, p. 31, 2019, doi: 10.3390/ijtpp4030031.



## New model

An equivalent formulation of the model can be derived from the **total energy equation**, leading to the following system of governing equations:
$$
\begin{gather}
\rho \frac{dv_m}{dm} + v_m  \frac{d \rho}{dm}
  = -\frac{\rho v_m}{A} \frac{dA}{dm} \\
\rho v_m \frac{dv_m}{dm} + \frac{dp}{dm}
  = \frac{\rho v_\theta^2}{r} \sin(\phi)
     - \frac{2 \tau}{b} \cos(\alpha)  \\
\rho v_m \frac{dv_\theta}{dm}
  = -\frac{\rho v_\theta v_m}{r} \sin(\phi)
     - \frac{2 \tau}{b} \sin(\alpha) \\
v_m \frac{dv_m}{dm} + v_\theta \frac{dv_\theta}{dm} + \frac{d h}{dm} 
  = \frac{2 \dot{q}}{b}
\end{gather}
$$
To close the system, the density is expressed as a function of pressure and enthalpy:
$$
 \frac{d \rho}{dm} = \left( \frac{\partial \rho}{\partial p} \right)_h  \frac{dp}{dm} + v_m \left( \frac{\partial \rho}{\partial h} \right)_p  \frac{d h}{dm}
$$
where the partial derivatives of density can be expressed in terms of the speed of sound and the Gruneisen parameter:
$$
\begin{align}
\left(\frac{\partial \rho}{\partial h}\right)_{p} = -\frac{\rho G}{c^2} \\[0.5em]
\left(\frac{\partial \rho}{\partial p}\right)_{h} = \frac{1 + G}{c^2}
\end{align}
$$
Substituting the above thermodynamic relations into the continuity equation gives the following quasi-linear system of ordinary differential equations:
$$
\begin{gather}
\frac{\rho}{v_m} \frac{dv_m}{dm} + \frac{1 + G}{c^2} \frac{dp}{dm} - \frac{\rho G}{c^2}  \frac{d h}{dm}
  = -\frac{\rho}{A} \frac{d A}{dm} \\
\rho v_m \frac{dv_m}{dm} + \frac{dp}{dm}
  = \frac{\rho v_\theta^2}{r} \sin(\phi)
     - \frac{2 \tau}{b} \cos(\alpha)  \\
\rho v_m \frac{dv_\theta}{dm}
  = -\frac{\rho v_\theta v_m}{r} \sin(\phi)
     - \frac{2 \tau}{b} \sin(\alpha) \\
v_m \frac{dv_m}{dm} + v_\theta \frac{dv_\theta}{dm} + \frac{d h}{dm} 
  = \frac{2 \dot{q}}{b}
\end{gather}
$$

### Characteristic determinant and singularity condition
This system of equations can be expressed in compact matrix form as:
$$
A\,\frac{dU}{dm} \;=\; S
$$
$$
A =
\begin{bmatrix}
\tfrac{\rho}{v_m} & 0 & -\,\tfrac{\rho G}{c^{2}} & \tfrac{1+G}{c^{2}} \\
\rho v_m & 0 & 0 & 1 \\
0 & \rho v_m & 0 & 0 \\
v_m & v_\theta & 1 & 0
\end{bmatrix},
\qquad
U =
\begin{bmatrix}
v_m\\
v_\theta\\
h\\
p
\end{bmatrix},
\qquad
S =
\begin{bmatrix}
-\tfrac{\rho}{A}\,\tfrac{dA}{dm} \\
\tfrac{\rho v_\theta^{2}}{r}\,\sin\phi - \tfrac{2\tau}{b}\cos\alpha  \\
-\tfrac{\rho v_\theta v_m}{r}\,\sin\phi - \tfrac{2\tau}{b}\sin\alpha \\
\tfrac{2\,\dot q_w}{b}
\end{bmatrix}.
$$
The determinant of the coefficient matrix $A$ is:
$$
\det(A) = \rho^{2} \left(1 - Ma_m^{2}\right), \qquad Ma_m = \frac{v_m}{a}
$$
This result shows that the system becomes singular when the meridional Mach number equals unity. At this condition, the flow reaches a sonic state along the mean streamline, corresponding to choking at the section of minimum area.


### Total pressure and entropy generation

To gain physical insight into aerodynamic losses, it is useful to analyze the mechanical energy equation and derive the corresponding expression for the stagnation pressure variation along the mean streamline.

Starting from the momentum equations, we multiply each by its respective velocity component:
$$
\begin{gather}
  \rho v_m \frac{d}{dm}\left(\frac{v_m^2}{2}\right) + v_m\frac{dp}{dm}
  = \frac{\rho v_\theta^2 v_m}{r} \sin(\phi)
     - \frac{2 \tau}{b} v_m \cos(\alpha)  \\[4pt]
\rho v_m \frac{d}{dm} \left(\frac{v_\theta^2}{2}\right)
  = -\frac{\rho v_\theta^2 v_m}{r} \sin(\phi)
     - \frac{2 \tau}{b} v_\theta \sin(\alpha)
\end{gather}
$$
Adding both equations and noting that $v_m = v \cos \alpha$ and $v_\theta = v \sin \alpha$, we obtain:
$$
v_m \left( \frac{dp}{dm} + \rho \frac{d}{dm}\left(\frac{v_m^2}{2} + \frac{v_\theta^2}{2} \right) \right)
= v_m \left( \frac{dp}{dm} + \rho v \frac{dv}{dm} \right)
= v_m \frac{dp_0}{dm}
= -\left(\frac{2}{b}\right) \tau v
$$
which can be rewritten as
$$
\frac{dp_0}{dm}  = -\frac{2 \tau}{b \cos \alpha}
$$

Recalling that the meridional coordinate $m$ is related to the actual streamline arclength $s$ by
$$
dm = \cos(\alpha)\, ds \to \frac{dp_0}{ds} = -\frac{2 \tau}{b}
$$
This relation shows explicitly that the loss of stagnation pressure arises from viscous shear stress $\tau$, that is, from forces that do not perform mechanical work in the energy equation. The factor $b$ is related to the hydraulic diameter of the channel, while the $\cos \alpha$ term accounts for the projection of the streamline direction onto the meridional plane. Consequently, total pressure loss scales with the actual streamline length, implying that longer flow paths or larger flow angles lead to greater viscous dissipation.

The total pressure loss can also be expressed in terms of the pressure loss coefficient $Y$ and the Fanning skin friction factor $c_f$:
$$
\begin{gather}
  Y = -\frac{\Delta p_0}{p_0 - p} \;\;\Rightarrow\;\; dY = -\frac{dp_0}{p_0 - p} \\[4pt]
  c_f = \frac{\tau}{\tfrac{1}{2}\rho v^2} \;\equiv\; \frac{\tau}{p_0 - p}
\end{gather}
$$
Substituting the total pressure gradient relation into these definitions gives
$$
\frac{dY}{ds} = -\,\frac{2}{b}\,c_f,
$$
or, upon integration along the flow path,
$$
Y = 2\,c_f\,\left(\frac{L}{b}\right).
$$
This formulation relates the nondimensional total pressure loss directly to the wall friction coefficient and the passage geometry, where $L$ is the total streamline length and $b$ the local channel width.

The total pressure drop can also be related directly to the **entropy generation** in the channel. Starting from the Gibbs relation, the definition of total pressure, and the energy equation:
$$
T\,ds = dh - \frac{dp}{\rho}, 
\qquad
dp_0 = dp + \rho v\,dv,
\qquad
dh + v\,dv = \frac{2}{b}\dot{q},
$$
combining these gives
$$
\rho T\,ds = -\,dp_0 + \frac{2}{b}\dot{q}.
$$
For adiabatic flow $(\dot{q} = 0)$1, this simplifies to
$$
T\,ds = -\,\frac{dp_0}{\rho},
$$
showing that the entropy generation along the flow path is directly proportional to the loss of total (stagnation) pressure. In general, any term that appears in the momentum equations but does not contribute to mechanical work in the total energy equation represents a source of irreversibility.  



## Modeling the flow losses
Following Aungier, viscous losses in vaneless passages can be represented as the combined effect of three mechanisms:
- Skin friction loss caused by wall shear stresses that dissipate mechanical energy into heat.  
- Diffusion loss associated with adverse pressure gradients during flow deceleration, leading to boundary-layer growth and increased energy dissipation.  
- Curvature loss resulting from boundary-layer distortion and secondary flow formation in curved passages such as crossover bends or return channels.

Each of these mechanisms contributes to the overall total pressure loss and, consequently, to entropy generation within the flow passage.

### Distribution of losses between momentum components

In Aungier’s formulation, the skin friction losses appear in both the meridional and tangential momentum equations, while the diffusion and curvature losses are applied only to the meridional component. As I see it, there is no strong physical justification for this asymmetry, and it likely affects the predicted flow angle distribution within the passage.

I believe that a more suitable approach is to apply all three loss contributions to both the meridional and tangential momentum equations. In this case, the effects can be lumped into a single equivalent shear stress term $\tau = \tfrac{1}{2}c_f \rho v^2$, which consistently influences both momentum components.

The total loss contribution can be expressed in terms of an overall sking friction coefficient:
$$
c_f = \sum_i c_{f,i} = c_{f,W} + c_{f,D} + c_{f,C}
$$
where:
- $c_{f,W}$ is the wall friction coefficient, computed using Reynolds number correlations for flows in straight pipes.  
- $c_{f,D}$ represents the diffusion loss coefficient, defined as
  $$
  c_{f,D} = D (1 - E)
  $$
  with
  $$
  D = \frac{b}{A} \frac{dA}{dm}, 
  \qquad 
  D_m = 0.4 \cos \alpha_{\text{in}} \left(\frac{b_{\text{in}}}{L_m}\right)^{0.35}
  $$
  and the diffusion efficiency $E$ given by
  $$
  E =
  \begin{cases}
  1, & D \le 0, \\[4pt]
  1 - 0.2 \left(\frac{D}{D_m}\right)^{2}, & 0 < D < D_m, \\[6pt]
  0.8 \sqrt{\frac{D_m}{D}}, & D \ge D_m
  \end{cases}
  $$

- $c_{f,C}$ corresponds to the curvature loss coefficient, expressed as

  $$
  c_{f,C} = \frac{1}{26} \, b\,\kappa\cos \alpha
  $$

  where $\kappa$ is the local curvature of the streamline.


## Expressing the loss distribution

The pressure recovery coefficient of a diffuser can be expressed as:
$$
C_p = \frac{p_2 - p_1}{p_{01}- p_1} = \frac{(p_{01}- p_1)- (p_{02}- p_2) - (p_{01}- p_{02}) }{p_{01}- p_1} = 1 - \frac{p_{02}- p_{2}}{p_{01}- p_1} - \frac{p_{01}- p_{02}}{p_{01}- p_1} 
$$
$$
C_p = 1 - \Delta C_{p,kinetic} - \Delta C_{p,loss}
$$

In an analogous way, the efficiency of a diffuser can be expressed as:
$$
\eta = \frac{h_{2s} - h_1}{h_{01}- h_1} = \frac{(h_{01}- h_1) - (h_{01}- h_{2}) - (h_{2}- h_{2s})}{h_{01}- h_1} = 1 - \frac{h_{01}- h_{2}}{h_{01}- h_1} - \frac{h_{2}- h_{2s}}{h_{01}- h_1} 
$$
$$
\eta = 1 - \Delta \eta_{p,kinetic} - \Delta \eta_{p,loss}
$$

These value of pressure recovery or efficiency might be higher than unity if the flow is not adiabatic.


## Thoughts for the paper

If there is time, we can explain the model for the 90--degree bend including first the flow equations explaining that they where modified for p-h function calls (in agremeent with look-up tables)

Then explain the geometry.

Simply phi=90 and m=r for the case of radial

for the 90-degree bent we have to define the equations of the NURBS

explain the geometric construction of the shape

3-degree bs-pline for the midline defined by parameters

Also a b-spline (linear) for the thickness distribution.

Explain the equations for the construction and arclength/coordinate reparametrization in an appendix?

