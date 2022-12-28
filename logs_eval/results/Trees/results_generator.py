import numpy as np

results = []

results.append([16.434492206033905, 0.76, 0.24, 119.78947368421052])
results.append([9.217483294028693, 0.7, 0.3, 123.08571428571429])
results.append([9.431210764063954, 0.72, 0.28, 126.5])

results.append([24.151430446779333, 0.98, 0.02, 125.91836734693878])
results.append([20.71481445587884, 1, 0, 131.98])
results.append([17.80080238090519, 0.96, 0.04, 136.75])

np.save('eval_50_SimpleAvoid_SimpleAvoid_SimpleMultirotor', np.array(results))
