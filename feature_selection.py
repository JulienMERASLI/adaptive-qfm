import numpy as np
from scipy import optimize
from scipy.optimize import milp



def convert_dicts_to_arrays(importances_dict, times_dict):
	feature_names = []
	importances = []
	times = []

	for feature_name in importances_dict:
		feature_names.append(feature_name)
		importances.append(importances_dict[feature_name])
		times.append(times_dict[feature_name])

	return feature_names, np.array(importances), np.array(times)

def select_features_from_performance_required(performance_required, importances_dict, times_dict):
	feature_names, importances, times = convert_dicts_to_arrays(importances_dict, times_dict)

	if (np.sum(importances) != 1):
		raise ValueError("The sum of importances must be equal to 1.")
	if (performance_required < 0 or performance_required > 1):
		raise ValueError("The performance required must be between 0 and 1. (percentage of maximum performance)")
	bounds = optimize.Bounds(0, 1)  # 0 <= x_i <= 1
	integrality = np.full_like(times, True, dtype=int)
	constraints = optimize.LinearConstraint(A=importances, lb=0, ub=1-performance_required)

	res = milp(c=-times, constraints=constraints,
           integrality=integrality, bounds=bounds)

	return {feature_names[i]: not bool(res.x[i]) for i in range(len(feature_names))}	


def select_features_from_time_budget(time_budget, importances_dict, times_dict):
	feature_names, importances, times = convert_dicts_to_arrays(importances_dict, times_dict)
	
	if (np.sum(importances) != 1):
		raise ValueError("The sum of importances must be equal to 1.")
	if (time_budget < 0 or time_budget > 1):
		raise ValueError("The time budget must be between 0 and 1. (percentage of maximum time)")
	bounds = optimize.Bounds(0, 1)  # 0 <= x_i <= 1
	integrality = np.full_like(importances, True, dtype=int)
	constraints = optimize.LinearConstraint(A=times, lb=0, ub=time_budget)

	res = milp(c=-importances, constraints=constraints,
			integrality=integrality, bounds=bounds)

	return {feature_names[i]: res.x[i] for i in range(len(feature_names))}	