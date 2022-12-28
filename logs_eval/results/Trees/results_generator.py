import numpy as np

results = []

results.append([30.056801369453236, 0.98, 0.02, 167.55102040816325])
results.append([35.29027739590203, 1, 0, 159.78])
results.append([26.644441169876337, 1, 0, 176.02])

results.append([29.622984700713218, 1, 0, 165.04])
results.append([24.766614548715282, 0.98, 0.02, 167.42857142857142])
results.append([27.566304489823043, 1, 0, 163.66])

np.save('eval_50_Trees_Trees_SimpleMultirotor', np.array(results))
