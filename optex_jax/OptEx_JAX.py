import math
import time
import numpy as np
from copy import deepcopy
from multiprocessing.pool import ThreadPool as Pool
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import optax
import jax

import warnings
warnings.filterwarnings("ignore")

class Greedy:
    def __init__(self, num_actions) -> None:
        self.num_actions = num_actions
        self.weights = np.ones(num_actions)
    
    def update_weights(self, rewards):
        self.weights = np.array(rewards)
        
    def select_action(self):
        return np.argmax(self.weights)
    
class MovingGreedy:
    def __init__(self, num_actions) -> None:
        self.num_actions = num_actions
        self.weights = np.zeros(num_actions)
        self.gama = 0.0
    
    def update_weights(self, rewards):
        self.weights = (1-self.gama) * np.array(rewards) + self.gama * self.weights
        
    def select_action(self):
        print(self.weights)
        return np.argmax(self.weights)


class Exp3:
    def __init__(self, num_actions) -> None:
        self.num_actions = num_actions
        self.weights = np.ones(num_actions)
        self.gamma = 0.0
    
    # 在每个时间步中，根据权重向量计算概率分布
    def compute_probs(self):
        weights_sum = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / weights_sum) + self.gamma / self.num_actions
        probs /= probs.sum()
        return probs

    # 根据概率分布选择一个动作
    def select_action(self):
        probs = self.compute_probs()
        return np.random.choice(self.num_actions, p=probs)

    # 更新权重向量
    def update_weights(self, rewards):
        probs = self.compute_probs()
        rewards = np.array(rewards)
        rewards = rewards / np.sum(rewards)
        print(rewards)
        
        xs = self.gamma / (probs.shape[0] * probs)
        self.weights = self.weights * np.exp(rewards * xs)
    
    # action = select_action(probabilities, subkey)


def tuning_mattern(xs, ys, target_xs, target_ys, choice=[0.1, 1, 10, 100], std=0.001, effective_dim=-1):
    loss = []
    n, d = xs.shape
    
    if effective_dim > 0:
        indices = np.random.choice(d, size=min(d, effective_dim), replace=False)    
        xs = xs[:,indices]
        target_xs = target_xs[:,indices]
        
    norm = np.sqrt(np.linalg.norm(xs, axis=-1)).mean()
    xs /= norm
    target_xs /= norm
    
    for l in choice:
        kernel = Matern(length_scale=l, nu=2.5)
        
        K_mat = kernel(xs, xs)
        K_mat_inv = np.linalg.inv(K_mat + std * np.eye(n))
        k_vec = kernel(target_xs, xs)
        pred_ys = jax.numpy.matmul(jax.numpy.matmul(k_vec, K_mat_inv), ys)
        
        loss += [jax.numpy.linalg.norm(pred_ys - target_ys)]
        
    return choice[np.argmin(loss)]

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2 + 1e-5)
    return similarity

def get_proxy_grad_func(x_values, y_values, kernel=1.0 * RBF(1.0), std=0.01, normalize_x=True, effective_dim=-1):
    n, d = x_values.shape
    
    # to check a smaller dimension can also help the optimzation
    if effective_dim > 0:
        indices = np.random.choice(d, size=min(d, effective_dim), replace=False)    
        x_values = x_values[:,indices]
        
    if normalize_x:
        norm = np.sqrt(np.linalg.norm(x_values, axis=-1)).mean()
        x_values /= norm
    
    K_mat_inv = np.linalg.inv(kernel(x_values, x_values) + std * np.eye(n))
    # K_mat_inv = np.linalg.pinv(kernel(x_values, x_values))
    ys = jax.numpy.einsum('bi,ij->bj', K_mat_inv, y_values)
    
    def proxy_grad_func(x):        
        x = x[indices] if effective_dim > 0 else x
        x = x / norm if normalize_x else x
        pred = jax.numpy.einsum('bi,ij->bj', kernel(x.reshape(1,-1), x_values), ys)
        return pred.reshape(-1)
    return proxy_grad_func


def run_standard(func, opt_name, lr, x0, num_iters, num_parall, datas=None, opt_state=None):
    
    # grad_func = jax.jit(jax.grad(func))
    fgx_func = jax.jit(jax.value_and_grad(func))
    global_opt = eval(opt_name)(learning_rate=lr)
    global_opt_state = global_opt.init(x0) if opt_state is None else opt_state
    
    x = x0
    
    for i in range(num_iters * num_parall):
        fx, grad = fgx_func(x) if datas is None else fgx_func(x, *datas[i])
        updates, global_opt_state = global_opt.update(grad, global_opt_state)
        x = optax.apply_updates(x, updates)
        
        # if i % 5 == 4 or i == 0:
        #     print("===>", i, "%.4f" % fx)
        # print("===>", i, "%.4f" % fx)
    
    return x, fx, global_opt_state


def run_line_search(func, opt_name, lr, x0, num_iters, num_parall, datas=None, opt_state=None, inter_results={}):
    # grad_func = jax.jit(jax.grad(func))
    fgx_func = jax.jit(jax.value_and_grad(func))
    global_opt = eval(opt_name)(learning_rate=lr)
    global_opt_state = global_opt.init(x0) if opt_state is None else opt_state
    
    def run_update(j, proxy_grad, global_opt_state, x, data=None):
        proxy_opt = eval(opt_name)(learning_rate=j * lr)
        proxy_updates, proxy_opt_state = proxy_opt.update((j > 0) * proxy_grad, global_opt_state)
        proxy_x = optax.apply_updates(x, proxy_updates)
        
        fx, grad = fgx_func(proxy_x) if data is None else fgx_func(proxy_x, *data)
        updates, proxy_opt_state = global_opt.update(grad, proxy_opt_state)
        x_update = optax.apply_updates(proxy_x, updates)
        return fx, proxy_x, grad, proxy_opt_state, x_update
    
    x = x0
    proxy_grad = jax.numpy.zeros_like(x) if "proxy_grad" not in inter_results else inter_results["proxy_grad"]
    
    for i in range(num_iters):
        caches = list(map(
            run_update, range(num_parall), [proxy_grad] * num_parall, [global_opt_state] * num_parall, [x] * num_parall, datas[i*num_parall:(i+1)*num_parall]
            ))
        idx = np.argmin([c[0] for c in caches]).item()
        # idx = np.argmin([jax.numpy.linalg.norm(c[2]) for j, c in enumerate(caches)]).item()
        fx, _, proxy_grad, global_opt_state, x = caches[idx]
        # print("===>", i, idx, "%.4f" % fx)
    
    inter_results.update({
        "proxy_grad": proxy_grad,
    })
    
    return x, fx, global_opt_state


def run_optex(func, opt_name, lr, x0, num_iters, num_parall, datas=None, opt_state=None, effective_dim=-1, inter_results={}):
    
    # grad_func = jax.jit(jax.grad(func))
    fgx_func = jax.jit(jax.value_and_grad(func))
    global_opt = eval(opt_name)(learning_rate=lr)
    global_opt_state = global_opt.init(x0) if opt_state is None else opt_state
    
    def run_proxy_update(opt_name, lr, proxy_grad_func, global_opt_state, x):
        # proxy_grad_func = jax.jit(proxy_grad_func)
        proxy_opt = eval(opt_name)(learning_rate=lr)
        proxy_x, proxy_opt_state = x, deepcopy(global_opt_state)
        
        proxy_x_cache = [proxy_x]
        proxy_opt_state_cache = [proxy_opt_state]
        
        for k in range(num_parall-1):
            proxy_grad = proxy_grad_func(proxy_x) # grad = grad_func(proxy_x)
            proxy_updates, proxy_opt_state = proxy_opt.update(proxy_grad, proxy_opt_state)
            proxy_x = optax.apply_updates(proxy_x, proxy_updates)
            
            proxy_x_cache.append(proxy_x)
            proxy_opt_state_cache.append(proxy_opt_state)
        
        return proxy_x_cache, proxy_opt_state_cache
    
    def run_parallelized_iterations(proxy_x, proxy_opt_state, data=None):
        fx, grad = fgx_func(proxy_x) if data is None else fgx_func(proxy_x, *data)
        updates, proxy_opt_state = global_opt.update(grad, proxy_opt_state)
        x_update = optax.apply_updates(proxy_x, updates)
        return fx, grad, proxy_opt_state, x_update
    
    x = x0
    
    if "x_history" not in inter_results:
        x_history, g_history = [], []
        selector = MovingGreedy(num_parall)
    else:
        x_history, g_history = inter_results["x_history"], inter_results["g_history"]
        selector = inter_results["selector"]
        
    if "length_scale" in inter_results.keys():
        length_scale = inter_results["length_scale"]
    else:
        length_scale = 0.1
    
    for i in range(num_iters):
        if len(x_history) < 2:
            proxy_grad_func = lambda z: jax.numpy.zeros_like(x)
        else:
            proxy_grad_func = get_proxy_grad_func(
                np.concatenate(x_history, axis=0),
                np.concatenate(g_history, axis=0),
                # kernel=gpjax.kernels.Matern52(lengthscale=64),
                kernel=1.0 * Matern(length_scale=length_scale, nu=2.5), 
                std=0.001,
                normalize_x=True,
                effective_dim=effective_dim
            )
        
        proxy_x_cache, proxy_opt_state_cache = run_proxy_update(opt_name, lr, proxy_grad_func, global_opt_state, x)
        
        caches = list(map(run_parallelized_iterations, proxy_x_cache, proxy_opt_state_cache, datas[i*num_parall:(i+1)*num_parall]))
        # caches = jax.pmap(run_parallelized_iterations)( proxy_x_cache, proxy_opt_state_cache, datas[i*num_parall:(i+1)*num_parall])
        # print("Elapsed Time of run_parallelized_iterations: %.4f" % (time.time() - start))
        # caches = list(map(run_parallelized_iterations, [proxy_x_cache[-1]], [proxy_opt_state_cache[-1]], [datas[(i+1)*num_parall - 1]]))
        
        # idx = np.argmin([c[0] for c in caches]).item()
        idx = -1
        # idx = np.argmin([jax.numpy.linalg.norm(c[1]) for c in caches]).item()
        fx, _, global_opt_state, x = caches[idx]
        print("===>", idx, "%.4f" % fx)
        
        x_history += [c.reshape(1,-1) for c in proxy_x_cache]
        g_history += [c[1].reshape(1,-1) for c in caches]
        # x_history += [proxy_x_cache[-1].reshape(1,-1)]
        # g_history += [caches[-1][1].reshape(1,-1)]
        
        # x_history = x_history[-20:]
        # g_history = g_history[-20:]
        
        # x_history = x_history[-20:]
        # g_history = g_history[-20:]
        
        x_history = x_history[-50:]
        g_history = g_history[-50:]
        
        print(len(x_history), len(g_history))

    inter_results.update({
        "x_history": x_history,
        "g_history": g_history,
        "selector": selector,
    })
    
    return x, fx, global_opt_state


def run_benchmark(func, opt_name, lr, x0, num_iters, num_parall, datas=None, opt_state=None, inter_results={}):
    # grad_func = jax.jit(jax.grad(func))
    fgx_func = jax.jit(jax.value_and_grad(func))
    
    global_opt = eval(opt_name)(learning_rate=lr)
    global_opt_state = global_opt.init(x0) if opt_state is None else opt_state
    
    def run_proxy_update(opt_name, lr, proxy_grad_func, global_opt_state, x, datas=None):
        proxy_opt = eval(opt_name)(learning_rate=lr)
        proxy_x, proxy_opt_state = x, deepcopy(global_opt_state)
        
        proxy_x_cache = [proxy_x]
        proxy_opt_state_cache = [proxy_opt_state]
        
        for k in range(num_parall-1):
            fx, proxy_grad = proxy_grad_func(proxy_x) if datas is None else proxy_grad_func(proxy_x, *datas[k])
            proxy_updates, proxy_opt_state = proxy_opt.update(proxy_grad, proxy_opt_state)
            proxy_x = optax.apply_updates(proxy_x, proxy_updates)
            
            proxy_x_cache.append(proxy_x)
            proxy_opt_state_cache.append(proxy_opt_state)
        return proxy_x_cache, proxy_opt_state_cache
        
    def run_parallelized_iterations(proxy_x, proxy_opt_state, data=None):
        fx, grad = fgx_func(proxy_x) if data is None else fgx_func(proxy_x, *data)
        updates, proxy_opt_state = global_opt.update(grad, proxy_opt_state)
        x_update = optax.apply_updates(proxy_x, updates)
        return fx, grad, proxy_opt_state, x_update
    
    x = x0
    
    if "selector" not in inter_results:
        selector = Exp3(num_parall)
        # selector = Greedy(num_parall)
    else:
        selector = inter_results["selector"]
    
    for i in range(num_iters):
        proxy_x_cache, proxy_opt_state_cache = run_proxy_update(
            opt_name, lr, fgx_func, global_opt_state, x, 
            None if datas is None else datas[i*num_parall:(i+1)*num_parall]
        )
        
        caches = list(map(run_parallelized_iterations, proxy_x_cache, proxy_opt_state_cache, datas[i*num_parall:(i+1)*num_parall]))
        # idx = np.argmin([func(x1, *x2) for x1, x2 in zip(proxy_x_cache, datas[i*num_parall:(i+1)*num_parall])]).item()
        # idx = np.argmin([jax.numpy.linalg.norm(c[0]) for c in caches]).item()
        # print("===>", i, idx, "%.4f" % func(caches[idx][-1], *datas[i*num_parall+idx]))
        idx = -1
        fx, _, global_opt_state, x = caches[idx]
        # print("===>", i, idx, "%.4f" % fx)
    
    inter_results.update({
        "selector": selector,
    })
    
    return x, fx, global_opt_state
    