import numpy as np

results = []

results.append([294.9498467246312, 0.64, 0.34, 316.59375])
results.append([336.4001846640506, 0.74, 0.24, 336.27027027027026])
results.append([350.63767655345924, 0.84, 0.16, 340.07142857142856])

np.save('eval_50_NH_center_NH_center_SimpleMultirotor', np.array(results))
