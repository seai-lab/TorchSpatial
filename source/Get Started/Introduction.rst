Introduction
+++++++++++++++++++++++

TorchSpatial is a PyTorch package for Spatial Representation Learning. Currently, it provides 15 commonly recognized location encoders, including location encoders built on projected 2D space and 3D space.
It is flexible enough to support the development of any location encoders. In the future, we plan also to support other spatial data types such as polylines, polygons, spatial networks, etc. 

2D Location Encoders
=====================

For location encoders applicable in a projected 2D space, TorchSpatial implements the following 7 models:

* *tile* is a discretized-based location encoder employed by many previous studies, which splits geographic areas into grids and uses grid embedding to represent locations within each grid. See details in :doc:`/2D Location Encoders/tile-ffn`.

* *wrap* uses a coordinate wrap mechanism to convert each dimension of location into 2 numbers and feed them into :math:`\mathbf{NN}^{\mathit{wrap}}()`.  :math:`\mathbf{NN}^{\mathit{wrap}}()` consists of four residual blocks which are implemented as linear layers. See details in :doc:`/2D Location Encoders/wrap`.

* *wrap + ffn* is a variant of *wrap* that substitutes :math:`\mathbf{NN}^{\mathit{wrap}}()` in *wrap* with :math:`\mathbf{NN}^{\mathit{ffn}}()`. See details in :doc:`/2D Location Encoders/wrap-ffn`.

* *rbf* randomly samples :math:`W` points from the training dataset as RBF anchor points, and uses Gaussian kernels :math:`\exp\left(-\frac{\lVert x_i - x_{anchor}\rVert_2^2}{2\sigma^2}\right)` on each anchor points, where :math:`\sigma` is the kernel size. Each input point :math:`x_i` is encoded as a :math:`W`-dimension RBF feature vector, which is fed into :math:`\mathbf{NN}^{\mathit{ffn}}()` to obtain the location embedding. See details in :doc:`/2D Location Encoders/rbf`.
    
* *rff* means *Random Fourier Features* . It first encodes location :math:`x` into a :math:`W` dimension vector - :math:`\mathit{PE}^{\mathrm{rff}}(x) = \phi(x) = \frac{\sqrt{2}}{\sqrt{W}}\bigcup_{i=1}^W\left[\cos\left(\omega_i^T x + b_i\right)\right]` where :math:`\omega_i \overset{\mathrm{i.i.d}}{\sim} \mathcal{N}(0, \delta^2 I)` is a direction vector whose each dimension is independently sampled from a normal distribution. :math:`b_i` is a shift value uniformly sampled from :math:`[0, 2\pi]` and :math:`I` is an identity matrix. Each component of :math:`\phi(x)` first projects :math:`x` into a random direction :math:`w_i` and makes a shift by :math:`b_i`. Then it wraps this line onto a unit circle in :math:`\mathbb{R}^2` with the cosine function. :math:`\mathit{PE}^{\mathrm{rff}}(x)` is further fed into :math:`\mathbf{NN}^{\mathit{ffn}}()` to get a location embedding. See details in :doc:`/2D Location Encoders/rff`.

* *Space2Vec-grid* and *Space2Vec-theory* are two multi-scale location encoder on 2D Euclidean space. Both of them implement the position encoder :math:`\mathit{PE}(x)` as a deterministic Fourier mapping layer which is further fed into the :math:`\mathbf{NN}^{\mathit{ffn}}()`. Both models' position encoders can be treated as performing a Fourier transformation on a 2D Euclidean space. See details in :doc:`/2D Location Encoders/Space2Vec-grid` and :doc:`/2D Location Encoders/Space2Vec-theory`.


3D Location Encoders
=====================

We also encompass 8 location encoders that learn location embeddings from 3D space as follows:
 
* *xyz* first uses position encoder :math:`\mathit{PE}^{\mathrm{xyz}}(x)` to convert the lat-lon spherical coordinates into 3D Cartesian coordinates centered at the sphere center. And then it feeds the 3D coordinates into a multilayer perceptron :math:`\mathbf{NN}^{\mathit{ffn}}()`. See details in :doc:`/3D Location Encoders/xyz`.

* *NeRF* can be treated as a multiscale version of *xyz* using Neural Radiance Fields (NeRF) for its position encoder. See details in :doc:`/3D Location Encoders/NeRF`.

* *Sphere2Vec-sphereC*, *Sphere2Vec-sphereC+*, *Sphere2Vec-sphereM*, *Sphere2Vec-sphereM+*, *Sphere2Vec-dfs* are variants of *Sphere2Vec*, a multi-scale location encoder for spherical surface. They are the first location encoder series that preserves the spherical surface distance between any two points to our knowledge. See details in :doc:`/3D Location Encoders/Sphere2Vec-sphereC`, :doc:`/3D Location Encoders/Sphere2Vec-sphereC+`, :doc:`/3D Location Encoders/Sphere2Vec-sphereM`, :doc:`/3D Location Encoders/Sphere2Vec-sphereM+`, and :doc:`/3D Location Encoders/Sphere2Vec-dfs`.

* *Siren(SH)* is also a spherical location encoder proposed recently. It uses spherical harmonic basis functions as the position encoder :math:`\mathit{PE}^{\mathit{Spherical-Harmonics}}(x)` and a sinusoidal representation network (SirenNets) as the :math:`\mathbf{NN}()`. See details in :doc:`/3D Location Encoders/Spherical-Harmonics`.
