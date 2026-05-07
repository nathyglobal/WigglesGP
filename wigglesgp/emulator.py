import pickle
import numpy as np

class SigmaEmulator:
    """
    GP emulator for the non-linear damping scale Sigma(z, omega).
    """

    def __init__(self, gp, metadata=None, h=0.67):
        self.gp = gp
        self.metadata = metadata or {}
        self.h = h
    
    @classmethod
    def train_sklearn(
        cls,
        X,
        y,
        yerr=None,
        kernel=None,
        normalise_y=True,
        n_restarts_optimizer=10,
        h=0.67,
        metadata=None,
    ):
        """
        Train a scikit-learn GaussianProcessRegressor for Sigma(z, omega).
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"X must have shape (n_samples, 2), got {X.shape}")
        
        if y.shape != (X.shape[0],):
            raise ValueError(f"y must have shape (n_samples,), got {y.shape}")
        
        if kernel is None:
            kernel = ConstantKernel(
                constant_value=1.0,
                constant_value_bounds=(1e-5, 1e5),
            ) * RBF(
                length_scale=[1.0, 1.0], 
                length_scale_bounds=(1e-5, 1e5),
            )

        if yerr is None:
            alpha = 1e-10
        else:
            yerr = np.asarray(yerr, dtype=float)
            if yerr.shape != y.shape:
                raise ValueError(f"yerr must have the same shape as y: got {yerr.shape} and {y.shape}")
            alpha = yerr**2

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=normalise_y,
            n_restarts_optimizer=n_restarts_optimizer,
        )
        gp.fit(X, y)

        metadata = dict(metadata or {})
        metadata.update(
            {
                "backend": "sklearn",
                "target": "sigma",
                "input_columns": ["z", "omega"],
                "kernel": str(gp.kernel_),
            }
        )
        return cls(gp=gp, metadata=metadata, h=h)
    

    def check_domain(self, z, omega):
        """
        Check whether z and omega lie inside the emulator training domain.
        """
        required = ["z_min", "z_max", "omega_min", "omega_max"]
        if not all(key in self.metadata for key in required):
            return

        z = np.asarray(z, dtype=float)
        omega = np.asarray(omega, dtype=float)

        z_min, z_max = self.metadata["z_min"], self.metadata["z_max"]
        omega_min, omega_max = self.metadata["omega_min"], self.metadata["omega_max"]

        if np.any((z < z_min) | (z > z_max)):
            raise ValueError(f"Requested z is outside the emulator training domain: [{z_min}, {z_max}]")
        
        if np.any((omega < omega_min) | (omega > omega_max)):
            raise ValueError(f"Requested omega is outside the emulator training domain: [{omega_min}, {omega_max}]")
        

    def predict_sigma(self, z, omega, return_std=False, check_domain=True):
        """
        Predict Sigma(z, omega)
        """

        z_arr, omega_arr = np.broadcast_arrays(
            np.asarray(z, dtype=float),
            np.asarray(omega, dtype=float),
        )

        if check_domain:
            self.check_domain(z_arr, omega_arr)

        X = np.column_stack((z_arr.ravel(), omega_arr.ravel()))
        pred = self.gp.predict(X, return_std=return_std)

        if return_std:
            mean, std = pred
            return mean.reshape(z_arr.shape), std.reshape(z_arr.shape)

        return pred.reshape(z_arr.shape)

    def damping(self, k, z, omega, return_sigma=False, check_domain=True):
        """
        Predict the Gaussian damping envelope D(k, z; omega).
        """
        k_arr = np.asarray(k, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        omega_arr = np.asarray(omega, dtype=float)

        k_grid, z_grid, omega_grid = np.broadcast_arrays(k_arr, z_arr, omega_arr)
        sigma = self.predict_sigma(
            z_grid, 
            omega_grid,
            check_domain=check_domain,
            )

        damping = np.exp(-0.5 * (self.h * k_grid * sigma) ** 2)

        if return_sigma:
            return damping, sigma
        
        return damping

    @classmethod
    def from_sigma_table(
        cls,
        path,
        feature_type=None,
        normalise_y=True,
        n_restarts_optimizer=10,
        h=0.67,
    ):
        """
        Train a SigmaEmulator from a fitted Sigma CSV table.
        """
        from .training import build_sigma_training_set, load_sigma_table

        z, omega, sigma, sigma_error = load_sigma_table(
            path,
            feature_type=feature_type,
        )

        X, y, yerr, metadata = build_sigma_training_set(
            z=z,
            omega=omega,
            sigma=sigma,
            sigma_error=sigma_error,
            feature_type=feature_type,
        )

        return cls.train_sklearn(
            X,
            y,
            yerr=yerr,
            normalise_y=normalise_y,
            n_restarts_optimizer=n_restarts_optimizer,
            h=h,
            metadata=metadata,
        )

    def to_file(self, path):
        """
        Save the emulator to disk.
        """
        state = {
            "gp": self.gp,
            "metadata": self.metadata,
            "h": self.h,
        }

        with open(path, "wb") as handle:
            pickle.dump(state, handle)
    
    @classmethod
    def from_file(cls, path):
        """
        Load a saved emulator from disk.
        """
        with open(path, "rb") as handle:
            state = pickle.load(handle)

        return cls(
            gp=state['gp'],
            metadata=state.get("metadata", {}),
            h=state.get("h", 0.67)
        )