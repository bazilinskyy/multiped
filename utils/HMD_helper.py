import math
import numpy as np
import pandas as pd
import os
import ast


class HMD_yaw():
    """Convert and aggregate Unity HMD quaternions into horizontal head heading."""

    @staticmethod
    def quaternion_to_unity_heading(w, x, y, z):
        """Return horizontal HMD heading for a Unity quaternion.

        Unity uses x right, y up, and z forward. Horizontal head turning is
        therefore rotation around the y-axis. Rather than extracting the
        conventional aerospace z-axis yaw, this method rotates Unity's forward
        vector ``(0, 0, 1)`` and measures its projection on the x-z ground
        plane.

        Parameters are scalar-first ``[w, x, y, z]`` because the CSV processing
        code reorders Unity's stored ``[x, y, z, w]`` columns before calling
        this method.

        Returns
        -------
        float
            Heading in radians in ``[-pi, pi]``. Zero points along Unity +z;
            positive values turn towards Unity +x. ``nan`` is returned when
            the quaternion is invalid or its forward direction has no stable
            horizontal projection.
        """
        q = np.asarray([w, x, y, z], dtype=float)
        if not np.all(np.isfinite(q)):
            return np.nan

        norm = np.linalg.norm(q)
        if norm <= np.finfo(float).eps:
            return np.nan
        w, x, y, z = q / norm

        # Third column of the quaternion rotation matrix: the rotated Unity
        # forward vector (0, 0, 1).
        forward_x = 2.0 * (x * z + w * y)
        forward_z = 1.0 - 2.0 * (x * x + y * y)

        if math.hypot(forward_x, forward_z) <= np.finfo(float).eps:
            return np.nan
        return math.atan2(forward_x, forward_z)

    def average_quaternions_eigen(self, quaternions):
        """
        Averages a list of quaternions using Markley's method via eigen decomposition.

        Markley's method (see: https://doi.org/10.2514/1.28949) computes the quaternion mean
        that minimises the sum of squared distances on the unit hypersphere, ensuring a robust
        and well-defined average for rotations.

        Args:
            quaternions (List[List[float]]): List of quaternions, each as [w, x, y, z].

        Returns:
            np.ndarray: The average quaternion as a numpy array [w, x, y, z].

        Raises:
            ValueError: If the input list is empty.

        Notes:
            - All input quaternions should be unit quaternions (normalised). This function will
            normalise them if they are not.
            - The sign of the output is chosen so the scalar component (w) is non-negative.
        """
        if len(quaternions) == 0:
            raise ValueError("No quaternions to average.")
        elif len(quaternions) == 1:
            return np.array(quaternions[0])

        # Convert to numpy array and ensure shape (N, 4)
        q_arr = np.array(quaternions)

        # Normalise each quaternion to unit length
        q_arr = np.array([q / np.linalg.norm(q) for q in q_arr])

        # Ensure quaternions are all in the same hemisphere
        # Flip quaternions with negative dot product to the first
        reference = q_arr[0]
        for i in range(1, len(q_arr)):
            if np.dot(reference, q_arr[i]) < 0:
                q_arr[i] = -q_arr[i]

        # Form the symmetric accumulator matrix
        A = np.zeros((4, 4))
        for q in q_arr:
            q = q.reshape(4, 1)  # Make column vector
            A += q @ q.T         # Outer product

        # Normalise by number of quaternions (optional)
        A /= len(q_arr)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        avg_q = eigenvectors[:, np.argmax(eigenvalues)]  # Pick eigenvector with largest eigenvalue

        # Ensure scalar-first order: [w, x, y, z]
        return avg_q if avg_q[0] >= 0 else -avg_q  # Normalise sign

    def compute_avg_yaw_from_matrix_csv(self, input_csv, output_csv=None, force=False):
        """
        Computes average horizontal Unity head heading for each timestamp.

        The method name and ``AvgYaw`` output column are retained for
        compatibility with the existing plotting pipeline. Their values now
        represent rotation around Unity's vertical y-axis, not conventional
        z-axis yaw.

        If output_csv is given and force=False and the file already exists,
        it is loaded and returned instead of recomputing.
        """

        # If we already have output & we're not forcing, just load and return
        if output_csv is not None and not force and os.path.isfile(output_csv):
            return pd.read_csv(output_csv)

        df = pd.read_csv(input_csv)
        participant_cols = [col for col in df.columns if col != "Timestamp"]

        results = []

        for _, row in df.iterrows():
            all_quats = []

            for col in participant_cols:
                try:
                    quats = ast.literal_eval(row[col])
                    if isinstance(quats, list) and len(quats) > 0:
                        all_quats.extend(quats)
                except Exception:
                    continue

            if all_quats:
                avg_quat = self.average_quaternions_eigen(all_quats)
                heading = self.quaternion_to_unity_heading(*avg_quat)
                results.append({"Timestamp": row["Timestamp"], "AvgYaw": heading})
            else:
                results.append({"Timestamp": row["Timestamp"], "AvgYaw": None})

        out_df = pd.DataFrame(results)

        if output_csv is not None:
            out_df.to_csv(output_csv, index=False)

        return out_df

