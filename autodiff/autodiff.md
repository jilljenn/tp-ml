% Automatic differentiation
% Jill-Jênn Vie
% Oct 6, 2023
---
aspectratio: 169
handout: true
header-includes: |
  ```{=tex}
  \usepackage{tikz}
  \usepackage{bm}
  \usepackage{booktabs}
  \def\bolda{{\bm{a}}}
  \def\boldth{{\bm{\theta}}}
  \def\X{{\bm{X}}}
  \def\x{{\bm{x}}}
  \def\y{{\bm{y}}}
  \def\Y{{\mathcal{Y}}}
  \def\W{{\bm{W}}}
  \def\b{{\bm{b}}}
  \def\p{{\bm{p}}}
  \def\L{\mathcal{L}}
  \def\R{\mathbb{R}}
  \def\th{\bm{\theta}}
  \def\diag{{\textnormal{diag}}}
  \def\softmax{{\textnormal{softmax}}}
  \def\softplus{{\textnormal{softplus}}}
  \def\sigmoid{{\textnormal{sigmoid}}}
  \def\logsumexp{{\textnormal{logsumexp}}}
  \usepackage[skins,minted]{tcolorbox}
  \definecolor{bgm}{rgb}{0.95,0.95,0.95}
  \newtcblisting{myminted}[3][]{listing engine=minted,listing only,#1,minted language=#3,colback=bgm,minted options={linenos,fontsize=\footnotesize,numbersep=2mm,escapeinside=||,mathescape=true}}
  ```
---

# Outline

:::::: {.columns}
::: {.column width=70%}
Data $\X \in \R^{n \times d}$ targets $\y \in \Y^n$

Model $f : \x \mapsto \hat{y} = f(\x)$

## Activation and loss $\L$

Why sigmoid? Why ReLU?

## Optimizers

Why stochastic gradient descent?

## Automatic differentiation

Why backpropagation? What does `loss.backward()` do?

## Unsupervised, semi-supervised, self-supervised

What if we have very few labels $\y$ or only $\X$?
:::
::: {.column width=30%}
![](tom.png)
\hfill \small \textcolor{gray}{@untitled01ipynb}
:::
::::::

# Optimization

Find the best parameters $\th$ to reach a goal: making $f_{\th}(x)$ close to $y$

Usually, minimize a differentiable loss function

## Regression: continuous target

Ex. $y \in \mathbb{R} \quad \mathcal{L} = \sum_i (\hat{y}_i - y_i)^2$ squared error

## Classification: discrete target

Ex. $y \in \{0, 1\}$

- Logistic regression $\hat{y} = \sigmoid(\bm{w}^T \bm{x}) \in \{0, 1\}$
- Logistic loss: $\L_i(p, y) = y \log p + (1 - y) \log (1 - p)$

Ex. $y \in \{1, \ldots, C\}$

- Multinomial regression $\hat{y} = \softmax(\W \x + b) \in \{0, 1\}^C$
- Cross-entropy loss: $\L_i(\p, y) = \sum_{c = 1}^C y_c \log p_c = \log p_y$

# Activation / link functions; why sigmoid, softmax?

![](figures/activation_functions.png)

\small \textcolor{gray}{(Ollion \& Grisel)}

\centering

\footnotesize
\begin{tabular}{ccc} \toprule
& Binary & Multiclass\\ \midrule
$f$ & $\softplus : x \mapsto \log (1 + \exp(x))$ & $\logsumexp^+ : \x \mapsto \log(1 + \sum_c \exp(x_c))$\\
$f'$ & $\sigmoid : x \mapsto 1 / (1 + \exp(-x))$ & $\softmax : \x \mapsto \exp(\x) / \sum_c \exp(x_c)$\\
$f''$ & $x \mapsto \sigmoid(x) (1 - \sigmoid(x))$ & $\x \mapsto \diag(s(\x)) - s(\x) s(\x)^T$\\ \bottomrule
\end{tabular}

# Gradient descent {.fragile}

\centering

![](figures/sgd.jpg){width=50%}

$\th_{t + 1} = \th_t - \gamma \nabla_{\th} \L$

\begin{myminted}{SGD}{python}
for each epoch:
    for |$\x$|, |$y$| in dataset:
        compute gradients |$\frac{\partial \L}{\partial \boldth}(f_\boldth(\x), y)$|  # also noted $\nabla_\boldth \L$
        |$\boldth \gets \boldth - \gamma \nabla_\boldth \L$|  # $\gamma$ is the learning rate
\end{myminted}

# SGD in PyTorch {.fragile}

\begin{myminted}{PyTorch SGD}{python}
import torch

# define model, n_epochs, trainloader
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for _ in range(n_epochs):
    for inputs, targets in training:
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
\end{myminted}

# Variants

## Batch gradient descent

Compute the gradient of loss on all examples and update parameters

\pause

## Gradient descent

For each example $\x_i, y$ update parameters

\pause

## Stochastic gradient descent

Sample examples $\x_i, y$ and update parameters

\pause

### Minibatch gradient descent

Sample a batch of examples, estimate full gradient from batch and update parameters

$$\th = \th - \gamma \frac{n}{B} \sum_{i \in B} \mathcal{L}(\x_i, y)$$

# Why SGD?

To minimize a function

Find the zeroes of its derivative

How to find the zeroes of a function?

# Newton's method: find $x$ such that $f(x) = 0$ {.fragile}

\centering

![](figures/newton.png){width=60%}

\raggedright

$f: \R \to \R$ differentiable, $\not\exists x, f'(x) = 0$

$$ x_{t + 1} = x_t - \frac{f(x_t)}{f'(x_t)} \qquad \parbox{0.45\textwidth}{\centering Quadratic convergence\\$\exists C > 0, |x_{t + 1} - \ell| \leq C |x_t - \ell|^2$}$$

# In higher dimension

Let $g : \R^n \to \R$ twice differentiable, with $n$ big

\pause

What is the size of $g'(\bm{x})$? Usually noted $\displaystyle \frac{\partial g}{\partial \bm{x}}$ or $\nabla_{\bm{x}} g$.

\pause

$$\x_{t + 1} = \x_t - \underbrace{g''(\x_t)^{-1}}_{\in \mathbf{R}^{n \times n},~O(n^3)} \underbrace{g'(\x_t)}_{\in \mathbf{R}^n}$$

# How to compute gradients automatically? What's in `loss.backward()`? {.fragile}

\centering\footnotesize
\begin{myminted}{PyTorch SGD}{python}
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
x = np.linspace(-7, 7, 200)
plt.plot(x, tanh(x),
      x, egrad(tanh)(x),                                     # first
      x, egrad(egrad(tanh))(x),                              # second
      x, egrad(egrad(egrad(tanh)))(x),                       # third
      x, egrad(egrad(egrad(egrad(tanh))))(x),                # fourth
      x, egrad(egrad(egrad(egrad(egrad(tanh)))))(x),         # fifth
      x, egrad(egrad(egrad(egrad(egrad(egrad(tanh))))))(x))  # sixth
plt.show()
\end{myminted}

\begin{center}\includegraphics[width=4cm]{tanh.png}\end{center}

# Existing methods and their limitations

## Numerical differentiation

$$\frac{f(x + h) - f(x)}h$$

Round-off errors

\pause

## Symbolic differentiation

Have to keep symbolic expressions at each step of the process

\pause

## Automatic differentiation

# Chain rule

$$ (f \circ g)' = g' \cdot (f' \circ g) $$

## Generalized chain rule

$$ \frac{df_1}{dx} = \frac{df_1}{df_2} \frac{df_2}{df_3} \cdots \frac{df_n}{dx} $$

# Reminder: Jacobians

Consider differentiable $f : \R^n \to \R^m$, its Jacobian $J_f \in \R^{m \times n}$ contains its first-order partial derivatives $\displaystyle (J_f)_{ij} = \frac{\partial f_i}{\partial x_j}$:

$$J_f ={\begin{bmatrix}{\dfrac {\partial f }{\partial x_{1}}}&\cdots &{\dfrac {\partial f }{\partial x_{n}}}\end{bmatrix}}={\begin{bmatrix}\nabla ^{\mathrm {T} }f_{1}\\\vdots \\\nabla ^{\mathrm {T} }f_{m}\end{bmatrix}}={\begin{bmatrix}{\dfrac {\partial f_{1}}{\partial x_{1}}}&\cdots &{\dfrac {\partial f_{1}}{\partial x_{n}}}\\\vdots &\ddots &\vdots \\{\dfrac {\partial f_{m}}{\partial x_{1}}}&\cdots &{\dfrac {\partial f_{m}}{\partial x_{n}}}\end{bmatrix}}$$

\pause

## Example: linear layer

$W$ is a $m \times n$ matrix

If $f(\x) = W \x$

$f_i = W_i \x = \sum_j W_{ij} x_j$ where $W_i$ is $i$th row of $W$

$\displaystyle (J_f)_{ij} = \frac{\partial f_i}{\partial x_j} = W_{ij}$ so $J_f = W$

# Generalized multivariate chain rule

Consider differentiable $f : \R^m \to \R^k, g : \R^n \to \R^m$ and $\bolda \in \R^n$.

$$D_\bolda (f \circ g) = D_{g(\bolda)} f \circ D_\bolda g$$

So the Jacobians verify:

$$ J_{f \circ g} = (J_f \circ g) J_g$$

# Computation graph

\centering
\begin{tikzpicture}[var/.style={draw,rounded corners=2pt}, every edge/.style={draw,->,>=stealth},xscale=2.5,yscale=2]
\node (x) [var] {$\x$};
\node (y) at (2.5,-1) [var] {$y$};
\node (W) at (0.5,-1) [var] {$\W$};
\node (b) at (1.5,-1) [var] {$\b$};
\node (z) at (1,0) [var] {$z$};
\node (f) at (2,0) [var] {$\softmax$};
\node[var] (loss) at (3,0) {$\L$};
\node (end) at (4,0) {};
\draw (x) edge (z);
\draw (W) edge (z);
\draw (b) edge (z);
\draw (z) edge node[above] {$z(\x)$} (f);
\draw (f) edge node[above] {$f(\x)$} (loss);
\draw (y) edge (loss);
\draw (loss) edge node[above] {$\L(f(\x), y)$} (end);
\end{tikzpicture}

$$\begin{aligned}
\frac{d\L}{db_1} & = \frac{d\L}{df} \frac{df}{db_1} = \frac{d\L}{df} \left( \frac{df}{dz} \frac{dz}{db_1} \right) \textnormal{ (forward)}\\
& = \frac{d\L}{dz} \frac{dz}{db_1} = \left( \frac{d\L}{df} \frac{df}{dz} \right) \frac{dz}{db_1} \textnormal{ (backward)}
\end{aligned}$$

Properly written: $J_{\L \circ \softmax \circ z} = (J_\L \circ f) (J_\softmax \circ z) J_z$

Given that $\x \in \R^d, z(\x), f(\x) \in \R^{d_2}, \L(f(\x), y) \in \R$,  
which order is better? \only<2>{This is why \alert{backpropagation}.}

# Reverse accumulation (in $\R$)

Let us note the adjoint $\bar{f} \triangleq \frac{d\L}{df}$.

\centering
\begin{tikzpicture}[var/.style={draw,circle}, every node/.style={minimum size=7mm}, every edge/.style={draw,->,>=stealth},xscale=2.5,yscale=2]
\node (f) [var] {$f$};
\node (end) at (1,0) {};
\node (u) at (-1,1) {};
\node (v) at (-1,-1) {};
\draw (u) edge[bend right,draw=none] coordinate[at start](u-b) coordinate[at end](f-b) (f)
          edge[bend left,draw=none] coordinate[at start](u-t) coordinate[at end](f-t) (f)
          (u-b) edge node[below] {$u$} (f-b);
\only<2>{\draw[red] (f-t) edge node[above right] {$\bar{f} \frac{df}{du}$} (u-t);}
\draw (v) edge[bend right,draw=none] coordinate[at start](v-b) coordinate[at end](f-bv) (f)
          edge[bend left,draw=none] coordinate[at start](v-t) coordinate[at end](f-tv) (f)
          (v-t) edge node[above] {$v$} (f-tv);
\only<2>{\draw[red] (f-bv) edge node[below right] {$\bar{f} \frac{df}{dv}$} (v-b);}
\draw (f) edge node[above] {$f(u,v)$} (end);
\only<2>{\draw[red] ([yshift=-2pt] end.west) edge node[below] {$\bar{f}$} ([yshift=-2pt] f.east);}
\end{tikzpicture}

# A complete example (Wikipedia)

![](figures/reverse-ad.png)

# Notebook: compute gradients

![](figures/autodiff.png)

# Computation graph: classifier

\centering
\begin{tikzpicture}[var/.style={draw,rounded corners=2pt}, every edge/.style={draw,->,>=stealth},xscale=2.5,yscale=2]
\node (x) [var] {$\x$};
\node (y) at (2.5,-1) [var] {$y$};
\node (W) at (0.5,-1) [var] {$\W$};
\node (b) at (1.5,-1) [var] {$\b$};
\node (z) at (1,0) [var] {$z$};
\node (f) at (2,0) [var] {$\softmax$};
\node[var] (loss) at (3,0) {$\L$};
\node (end) at (4,0) {};
\draw (x) edge (z);
\draw (W) edge (z);
\draw (b) edge (z);
\draw (z) edge node[above] {$z(\x)$} (f);
\draw (f) edge node[above] {$f(\x)$} (loss);
\draw (y) edge (loss);
\draw (loss) edge node[above] {$\L(f(\x), y)$} (end);
\end{tikzpicture}

Here we define $\L$ as cross-entropy:
$$ \L(f(\x), y) = - \sum_{c = 1}^K \mathbf{1}_{y = c} \log {f(\x)}_c = - \log {f(\x)}_y $$
Compute $\frac{d\L}{dz_c}$

# Unsupervised, semi-supervised, self-supervised

What if we have very few labels $\y$ or only $\X$?

# Next week: introduction to reinforcement learning

\footnotesize Dynamic programming's *Principle of Optimality* (Bellman, 1952)
$$ \left. \begin{array}{rr}
\textnormal{value function} & \alert{v_\pi(s)}\\
\textnormal{action-value function} & \alert{q_\pi(s, a)}
\end{array}\right\} \textnormal{ for policy } \pi: S \to A \to R \to S' \to A' $$
Q-learning (1992), SARSA (1996), policy gradient (2000)

\centering

![](rl.png){width=70%}
