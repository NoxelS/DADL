import pickle
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt

data = None
with open("klima_data.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array([int(round(d)) for d in data["pegel"]])
Y = np.array([int(round(d)) for d in data["niederschlag"]])

S = np.unique(X)
S_keys = np.arange(0, len(S))
S_dict = dict(zip(S, S_keys))

V = np.unique(Y)
V_keys = np.arange(0, len(V))
V_dict = dict(zip(V, V_keys))

# print(len(S_dict.keys())) # 100
# print(len(V_dict.keys())) # 22

# Zustandsübergangsmatrix
Delta = np.zeros((len(S_dict.keys()), len(S_dict.keys())))

for t in range(len(X) - 2):
    this_state = S_dict[X[t+1]]
    last_state = S_dict[X[t]]
    Delta[this_state, last_state] += 1

# Normalisieren
Delta /= (len(X) - 1)
plt.imsave("Delta.png", Delta)


# Beobachtungsmatrix
Lambda = np.zeros((len(S_dict.keys()), len(V_dict.keys())))
for p in X:
    for n in Y:
        s_index = S_dict[p]
        v_index = V_dict[n]
        Lambda[s_index, v_index] += 1

Lambda /= (len(X) - 1)
plt.imsave("Lambda.png", Lambda)

def forward(alpha_i_t, alpha_i, t, T, len_S, Delta, Lambda, fw):
    if fw:
        # print("Forward")
        dt = 1
    else:
        # print("Backward")
        dt = -1
    # print("-Algorithmus, t:", t)

    if fw and t == 0 or (not fw) and t == T:
        pi_i = np.zeros([1, len_S])
        pi_i[:, initial_state_index] = 1
        alpha_i = pi_i * Lambda[:, 0]

    if fw and t < T or (not fw) and t > 0:
        if t >= 1:
            alpha_i = (alpha_i @ Delta.transpose()) * Lambda[:, V_dict[Y[t]]]
            # print("FW:",t,"alpha_i:", alpha_i)
        alpha_i_t[:, t] = alpha_i
        forward(alpha_i_t, alpha_i, t + dt, T, len_S, Delta, Lambda, fw)

    if fw and t == T-1:
        P_gesamt = sum(alpha_i_t[:, -1])
        print("End of forward algorithm, P_gesamte:", P_gesamt)
        return P_gesamt


def plot_foward_coeffs(alpha_i_t, beta_i_t):
    p, (ax1, ax2) = plt.subplots(2,1)
    im1 = ax1.imshow(np.log(alpha_i_t))
    im2 = ax2.imshow(np.log(beta_i_t))
    bar = plt.colorbar(im1, location="bottom")
    bar.set_label("logarithm of forward coefficients")
    plt.show()

initial_state_index = S_dict[X[0]]
alpha_i_t = np.zeros([len(S), len(Y)])
forward(alpha_i_t, None, 0, len(Y), len(S), Delta, Lambda, True)

initial_state_index = S_dict[X[len(S)- 1]]
beta_i_t = np.zeros([len(S), len(Y)])
forward(beta_i_t, None, len(Y) - 1, len(Y) - 1, len(S), Delta, Lambda, False)

# plot_foward_coeffs(alpha_i_t, beta_i_t)

# Define the gamma_i_t matrix
gamma_i_t = np.zeros([len(S), len(Y)])
for t in range(len(Y)):
    gamma_i_t[:, t] = alpha_i_t[:, t] * beta_i_t[:, t]
    gamma_i_t[:, t] /= sum(gamma_i_t[:, t])

print("gamma_i_t:", gamma_i_t)