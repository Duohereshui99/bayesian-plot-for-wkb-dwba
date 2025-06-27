from mpmath import coulombf, coulombg
import numpy as np
import mpmath
import numba
from scipy.integrate import simpson
import matplotlib.pyplot as plt

from scipy.optimize import root_scalar
# 设置高精度计算（50位小数）
mpmath.mp.dps = 50
# # 计算 F 和 G
# F_values = [mpmath.coulombf(l, eta, rho) for rho in rho_values]
# G_values = [mpmath.coulombg(l, eta, rho) for rho in rho_values]

e2=1.43997 ; hbarc=197.3269718 ; amu=931.49432
z1=88 ; m1=213*amu+0.345557
z2=2 ; m2=4*amu+2.42491587
Q=9.43530713
mu=m1*m2/(m1+m2)
v0=162.3 ; a0=0.4 ; r0=7.672; l=5; P0=0.03
rc=r0
k=np.sqrt(2*mu*Q/(hbarc**2))
eta= z1 * z2 * e2 * mu / (hbarc**2 * k)

#potential function
@numba.njit
def wspot(r,v0,a0,r0):
    return -v0*(1+np.cosh(r0/a0))/(np.cosh(r/a0)+np.cosh(r0/a0))

@numba.njit
def vc(r,z1,z2,rc):
    return np.where(
        r < rc,
        z1 * z2 * e2 * (3 - r**2 / rc**2) /(2 * rc),
        z1 * z2 * e2 / r
    )
    
@numba.njit
def vpot(r,v0,a0,r0,z1,z2,rc,l):
    return wspot(r,v0,a0,r0)+vc(r,z1,z2,rc)+(l+0.5)**2/r**2*hbarc**2/(2*mu)


# \delta V_c=V_c-z1z2e2/r
@numba.njit
def deltaVc(r,v0,a0,r0,z1,z2,rc,l):
    return vpot(r,v0,a0,r0,z1,z2,rc,l)-z1*z2*e2/r

#coulomb function

def F_L(r, k, eta, L):
    """库仑函数 F_L(η, kr)"""
    return coulombf(L, eta, k * r)

def G_L(r, k, eta, L):
    """库仑函数 G_L(η, kr)"""
    return coulombg(L, eta, k * r)



#mesh
h = 0.01/k
n_mesh = 8000
r1 = 2*l*h
r = np.arange(r1, r1 + h * n_mesh, h)


FL = np.array([F_L(ri, k, eta, l) for ri in r], dtype=np.complex128)


# 定义计算 y(0) 的函数
def calculate_y_at_r0(v0, r, h, Q, a0, r0, z1, z2, rc, l, k, eta):
    z = np.zeros_like(r, dtype=np.complex128)
    y = np.zeros_like(r, dtype=np.complex128)
    
    # 边界条件
    z[-1] = G_L(r[-1], k, eta, l) + 1j * F_L(r[-1], k, eta, l)
    z[-2] = G_L(r[-2], k, eta, l) + 1j * F_L(r[-2], k, eta, l)
    
    # Numerov 方法
    for i in range(len(z)-2, 0, -1):
        x = r[i]
        z[i-1] = 2*z[i] - z[i+1] + h**2 * (2*mu/hbarc**2) * (vpot(x, v0, a0, r0, z1, z2, rc, l) - Q) * z[i]
    
    # 计算 trial wavefunction y
    f = 2*mu/hbarc**2 * (vpot(r, v0, a0, r0, z1, z2, rc, l) - Q)
    y = (1 - h**2 * f / 12) * z
    y = np.array(y, dtype=np.complex128)
    # 归一化
    # 计算模方 |y|^2
    y_squared = np.abs(y)**2
    # 归一化常数
    # 数值积分（Simpson）
    norm_factor = np.sqrt(simpson(y=y_squared, x=r))
    # 归一化
    y /= norm_factor
    # 返回 y(r=0)实部
    return  np.real(y[0])  # y[0] 对应 r=0（网格的第一个点）
# y[0] 对应 r=0.5（网格的第一个点）

# 优化目标函数
def objective(v0):
    return calculate_y_at_r0(v0, r, h, Q, a0, r0, z1, z2, rc, l, k, eta)
# 优化目标函数 - 寻找使y(0)=0的v0
def find_v0_for_zero_y0(v0_min, v0_max):
    # 使用root_scalar寻找使y(0)=0的v0
    result = root_scalar(
        lambda v0: calculate_y_at_r0(v0, r, h, Q, a0, r0, z1, z2, rc, l, k, eta),
        bracket=[v0_min, v0_max],
        method='brentq'
    )
    return result.root


# # 参数设置
# v0_min, v0_max = 150, 170  # 搜索范围

# #寻找使y(0)=0的v0值
# optimal_v0 = find_v0_for_zero_y0(v0_min, v0_max)
# # print(f"Optimal v0 where y(0)=0: {optimal_v0:.4f} MeV")

# y0_at_optimal_v0 = calculate_y_at_r0(v0, r, h, Q, a0, r0, z1, z2, rc, l, k, eta)
# print(f"y(0) at optimal v0: {y0_at_optimal_v0:.4f}")
# v0=optimal_v0
#封装dwba
def model(v0, a0, r0, z1, z2, rc, l, P0, Q):
    # 计算波函数y
    z = np.zeros_like(r, dtype=np.complex128)
    y = np.zeros_like(r, dtype=np.complex128)
    
    # 边界条件
    z[-1] = G_L(r[-1], k, eta, l) + 1j * F_L(r[-1], k, eta, l)
    z[-2] = G_L(r[-2], k, eta, l) + 1j * F_L(r[-2], k, eta, l)
    # Numerov方法
    f = 2*mu/hbarc**2 * (vpot(r, v0, a0, r0, z1, z2, rc, l) - Q)

    for i in range(len(z)-2, 0, -1):
        x = r[i]
        z[i-1] = 2*z[i] - z[i+1] + h**2 * f[i] * z[i]
    
    # 计算trial wavefunction y

    y = (1 - h**2 * f / 12) * z
    y = np.array(y, dtype=np.complex128)
    
    # 归一化
    y_squared = np.abs(y)**2
    norm_factor = np.sqrt(simpson(y=y_squared, x=r))
    y /= norm_factor

    # 计算FL和VV

    VV = deltaVc(r, v0, a0, r0, z1, z2, rc, l) - (l+0.5)**2/r**2*hbarc**2/(2*mu)
    
    # 计算积分结果
    result = simpson(y=VV*FL*y, x=r)
    
    # 计算半衰期
    Gamma = P0 * abs(result)**2 * 4*mu/hbarc**2/k
    T_half = hbarc * np.log(2) / Gamma * 1e-23/3
    
    return T_half  # 返回半衰期，单位为秒

# t_half = model(v0, a0, r0, z1, z2, rc, l, P0, Q)
# print(t_half)


# experiment data and corresponding sigma
y=np.array([254e-6])  # s
sigma=np.array([2e-6]) # s 

def log_prior(a):
    vv,aa,rr,pp=a
    R=1
    v0=162.3; a0=0.4; r0=7.672; P0=0.03
    
    sigma_v=2.5*R
    sigma_a=0.03*R
    sigma_r=0.05*R
    sigma_P=0.005*R

    prior_v=-0.5*(vv-v0)**2/sigma_v**2
    prior_a=-0.5*(aa-a0)**2/sigma_a**2
    prior_r=-0.5*(rr-r0)**2/sigma_r**2
    prior_P=-0.5*(pp-P0)**2/sigma_P**2
    return prior_v+prior_a+prior_r+prior_P

# def log_posterior(a,y,sigma,m1,m2,z1,z2,l,Q):
#     log_prior_value=log_prior(a)

#     f=model(a[0],a[1],a[2],z1,z2,a[2],l,a[3],Q)
#     if np.isnan(f) or np.isinf(f):
#             with open("invalid_params.txt", "a") as file:  # "a" 表示追加模式
#                 file.write(f"NaN/Inf in model output: {a}\n")
#             return -np.inf, None  # 模型输出非法，拒绝
#     # 计算对数似然
#     residual = (y - f) / sigma


#     log_likelihood = -0.5 * np.sum(residual**2)
#     total = log_likelihood + log_prior_value 
#     # 确保最终结果合法
#     return total if np.isfinite(total) else -np.inf, None   
def log_posterior(a, y, sigma, m1, m2, z1, z2, l, Q):
    log_prior_value = log_prior(a)
    
    if np.isinf(log_prior_value):
        raise ValueError("Invalid prior")
    
    f = model(a[0], a[1], a[2], z1, z2, a[2], l, a[3], Q)
    if np.isnan(f) or np.isinf(f):
        raise ValueError("Invalid model output")
    
    residual = (y - f) / sigma
    log_likelihood = -0.5 * np.sum(residual**2)
    total = log_likelihood + log_prior_value
    
    if not np.isfinite(total):
        raise ValueError("Non-finite posterior")
    return total


M=4
nwalkers=2*M
initial_pos = [v0, a0, r0, P0]
pertubation_scale=[0.1, 0.05, 0.1, 0.001]
a=np.array([initial_pos+np.random.normal(0,pertubation_scale,M) for _ in range(nwalkers)])
# print(a)



import emcee
import multiprocessing

def run_mcmc_strict_nan_rejection(sampler, initial_state, nsteps):
    state = initial_state.copy()
    for step in range(nsteps):
        while True:
            try:
                # 直接存储结果，不使用store=False
                state, _, _ = sampler.run_mcmc(state, 1)
                break
            except (ValueError, RuntimeWarning):
                continue
        if step % 1000 == 0:
            print(f"Completed {step}/{nsteps} steps")
    return state
# with multiprocessing.Pool() as pool:
#     sampler = emcee.EnsembleSampler(nwalkers, M, log_posterior, args=[y,sigma,z1,z2,l,Q], a=1, pool=pool)
#     state = sampler.run_mcmc(a, 200)
#     sampler.reset()
#     sampler.run_mcmc(state, 50000)


with multiprocessing.Pool() as pool:
    # 初始化采样器
    sampler = emcee.EnsembleSampler(
        nwalkers, 
        M, 
        log_posterior, 
        args=[y, sigma, m1, m2, z1, z2, l, Q],
        a=1.0,  # 步长参数
        pool=pool
    )
    
    # 第一阶段：Burn-in
    print("Running burn-in with strict NaN rejection...")
    state = run_mcmc_strict_nan_rejection(sampler, a, 100)
    
    # 重置采样器
    sampler.reset()
    
    # 第二阶段：正式采样
    print("Running production with strict NaN rejection...")
    state = run_mcmc_strict_nan_rejection(sampler, state, 15000)
    
    # 获取最终链
    chain = sampler.get_chain(flat=True)
    print("Sampling completed. Chain shape:", chain.shape)
    print("Acceptance fraction:", sampler.acceptance_fraction)

    np.save("mcmc_chain_flat_217Th.npy", chain)





import prettyplease

samples = sampler.get_chain(flat=True)

labels=["$V_0$","$a_0$","$r_0$","$P_0$"]
fig = prettyplease.corner(samples, labels=labels)
# plt.show()
plt.savefig('217Th.pdf') 


# predictions = np.array([Model(m1, m2, z1, z2, l, Q, a[0], a[1], a[2], a[3]) for a in samples], dtype=float)
# #所有sample对应的模型输出

from multiprocessing import Pool

def model_prediction(a):
    return model(a[0], a[1], a[2], z1, z2, a[2], l, a[3], Q)

with Pool() as pool:
    predictions = np.array(pool.map(model_prediction, samples), dtype=float)
    
# 检查 predictions 的类型
print(predictions.dtype)




#计算1sigma,2sigma
# 计算预测值的均值和标准差
mean_prediction = np.mean(predictions, axis=0)
std_prediction = np.std(predictions, axis=0)
# 计算中位数
median_prediction = np.median(predictions, axis=0)
# 计算误差条
errorbar_1sigma = std_prediction  # 1σ
errorbar_2sigma = 2 * std_prediction  # 2σ
# 输出结果
print("Mean Prediction:", mean_prediction)
print("Median Prediction:", median_prediction)
print("1σ Error Bar:", errorbar_1sigma)
print("2σ Error Bar:", errorbar_2sigma)