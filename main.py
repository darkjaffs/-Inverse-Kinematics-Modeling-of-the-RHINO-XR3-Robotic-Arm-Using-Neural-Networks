import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tqdm import tqdm

# === DH Parameters (actual) ===
d = np.array([26.04, 0.0, 0.0, 0.0, 16.83]) / 100
a = np.array([0.0, 22.86, 22.86, 0.95, 0.0]) / 100
alpha = np.array([-np.pi/2, 0, 0, np.pi/2, 0])

# === Joint Limits ===
theta_limits = np.array([
    [np.radians(-180), np.radians(180)],
    [np.radians(-45), np.radians(45)],
    [np.radians(-90), np.radians(90)],
    [np.radians(-90), np.radians(90)],
    [np.radians(-90), np.radians(90)]
])

# === FK Helper Function ===
def dh_transform(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

# === Step Size (Finer Sampling) ===
delta_theta = np.radians(15)

# === Joint Ranges ===
theta1 = np.arange(theta_limits[0, 0], theta_limits[0, 1] + delta_theta, delta_theta)
theta2 = np.arange(theta_limits[1, 0], theta_limits[1, 1] + delta_theta, delta_theta)
theta3 = np.arange(theta_limits[2, 0], theta_limits[2, 1] + delta_theta, delta_theta)
theta4 = np.arange(theta_limits[3, 0], theta_limits[3, 1] + delta_theta, delta_theta)
theta5 = np.arange(theta_limits[4, 0], theta_limits[4, 1] + delta_theta, delta_theta)

# === Max Sample Estimate ===
max_samples = len(theta1) * len(theta2) * len(theta3) * len(theta4) * len(theta5)
print(f"Maximum possible samples: {max_samples}")

# Preallocate arrays
X = np.zeros((max_samples, 12))  # 3 for position + 9 for rotation matrix
Y = np.zeros((max_samples, 5))   # 5 joint angles
index = 0

# === Generate Dataset ===
print("Generating dataset...")
# Using a reduced nested loop approach to make the code more manageable
# Taking a subset of the combinations to make the dataset generation faster
sample_rate = 5  # Sample every nth value

for t1 in tqdm(theta1[::sample_rate]):
    for t2 in theta2[::sample_rate]:
        for t3 in theta3[::sample_rate]:
            for t4 in theta4[::sample_rate]:
                for t5 in theta5[::sample_rate]:
                    thetas = np.array([t1, t2, t3, t4, t5])
                    T = np.eye(4)
                    for j in range(5):
                        T = np.matmul(T, dh_transform(thetas[j], d[j], a[j], alpha[j]))
                    
                    pos = T[0:3, 3]  # End-effector position
                    R = T[0:3, 0:3]  # Rotation matrix
                    X[index, :] = np.concatenate([pos, R.flatten()])
                    Y[index, :] = thetas
                    index += 1

# === Trim Excess ===
X = X[:index, :]
Y = Y[:index, :]

print(f"Total samples: {X.shape[0]}")

# === Normalize Inputs/Outputs ===
Xmean = np.mean(X, axis=0)
Xstd = np.std(X, axis=0)
Ymean = np.mean(Y, axis=0)
Ystd = np.std(Y, axis=0)

Xn = (X - Xmean) / Xstd
Yn = (Y - Ymean) / Ystd

# === Train/Test Split ===
nTrain = round(0.8 * Xn.shape[0])
Xtrain, Ytrain = Xn[:nTrain, :], Yn[:nTrain, :]
Xval, Yval = Xn[nTrain:, :], Yn[nTrain:, :]

# === Neural Network Definition ===
model = models.Sequential([
    layers.Input(shape=(12,)),
    layers.Dense(256),
    layers.ReLU(),
    layers.Dense(128),
    layers.ReLU(),
    layers.Dense(64),
    layers.ReLU(),
    layers.Dense(5)
])

# === Training Options ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse'
)

# Add callbacks for early stopping and learning rate reduction
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
]

# === Train Neural Network ===
history = model.fit(
    Xtrain, Ytrain,
    epochs=200,
    batch_size=64,
    validation_data=(Xval, Yval),
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid(True)
plt.show()

# === Test on a Random Sample ===
idx = np.random.randint(0, X.shape[0])
true_input = X[idx, :]
true_output = Y[idx, :]

# Normalize & Predict
test_input = (true_input - Xmean) / Xstd
test_input = np.expand_dims(test_input, axis=0)  # Add batch dimension
pred_norm = model.predict(test_input)[0]
pred_output = pred_norm * Ystd + Ymean

# === Results ===
print("\nTarget Angles (deg):")
print(np.degrees(true_output))
print("Predicted Angles (deg):")
print(np.degrees(pred_output))

# Save the model and normalization parameters
model.save('rhino_xr3_ik_model.keras')
np.savez('normalization_params.npz', Xmean=Xmean, Xstd=Xstd, Ymean=Ymean, Ystd=Ystd)

# Function to test the trained model with new end-effector positions
def predict_joint_angles(position, rotation_matrix):
    """
    Predict joint angles from end-effector position and rotation matrix
    
    Parameters:
    position - 3D position vector [x, y, z]
    rotation_matrix - 3x3 rotation matrix
    
    Returns:
    joint_angles - 5 joint angles in degrees
    """
    # Flatten inputs
    input_vector = np.concatenate([position, rotation_matrix.flatten()])
    
    # Normalize
    normalized_input = (input_vector - Xmean) / Xstd
    normalized_input = np.expand_dims(normalized_input, axis=0)
    
    # Predict
    pred_norm = model.predict(normalized_input)[0]
    joint_angles = pred_norm * Ystd + Ymean
    
    return np.degrees(joint_angles)