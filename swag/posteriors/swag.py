"""
    implementation of SWAG
"""

import torch
import numpy as np
import itertools
from torch.distributions.normal import Normal
import copy

# import gpytorch
# from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
# from gpytorch.distributions import MultivariateNormal

from ..utils import flatten, unflatten_like

import tensorflow as tf 
import tensorflow_probability as tfp 
MultivariateNormal=tfp.distributions.MultivariateNormalFullCovariance

def swag_parameters(model, no_cov_mat=True):
    # for name in list(module._parameters.keys()):
    #     if module._parameters[name] is None:
    #         continue
    #     data = module._parameters[name].data
    #     module._parameters.pop(name)
    #     module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
    #     module.register_buffer("%s_sq_mean" % name, data.new(data.size()).zero_())

    #     if no_cov_mat is False:
    #         module.register_buffer(
    #             "%s_cov_mat_sqrt" % name, data.new_empty((0, data.numel())).zero_()
    #         )

    #     params.append((module, name))
    params = dict()
    names = list()
    #print(model.trainable_variables)
    for var in model.trainable_variables:
        name = var.name
        names.append(name)
        if var is None:
            continue
        # data = var
        # module._parameters.pop(name)
        # module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
        # module.register_buffer("%s_sq_mean" % name, data.new(data.size()).zero_())

        params["%s_mean"% name] = tf.zeros_like(var)
        params["%s_sq_mean"% name] = tf.zeros_like(var)


        if no_cov_mat is False:
            # module.register_buffer(
            #     "%s_cov_mat_sqrt" % name, data.new_empty((0, data.numel())).zero_()
            # )
            numel=1
            for s in var.shape:
                numel*=s
            params["%s_cov_mat_sqrt" % name] = tf.zeros(shape=(1, numel))
    return params, names
        # params.append((module, name))


class SWAG(tf.keras.Model):
    def __init__(
        self, base, no_cov_mat=True, max_num_models=0, var_clamp=1e-30, *args, **kwargs
    ):
        super(SWAG, self).__init__()

        #self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        self.n_models = tf.zeros(shape=(1))
        #self.params = list()
        self.params, self.names = swag_parameters(base, no_cov_mat)

        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models

        self.var_clamp = var_clamp

        #self.base = base(*args, **kwargs)
        self.base = base
        # self.base.apply(
        #     lambda module: swag_parameters(
        #         module=module, params=self.params, no_cov_mat=self.no_cov_mat
        #     )
        # )

    def call(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def sample(self, scale=1.0, cov=False, seed=None, block=False, fullrank=True):
        if seed is not None:
            #torch.manual_seed(seed)
            tf.random.set_seed(seed)

        if not block:
            self.sample_fullrank(scale, cov, fullrank)
        else:
            self.sample_blockwise(scale, cov, fullrank)

    def sample_blockwise(self, scale, cov, fullrank):
        # for module, name in self.params:
        #     mean = module.__getattr__("%s_mean" % name)

        #     sq_mean = module.__getattr__("%s_sq_mean" % name)
        #     eps = torch.randn_like(mean)

        #     var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

        #     scaled_diag_sample = scale * torch.sqrt(var) * eps

        #     if cov is True:
        #         cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
        #         eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
        #         cov_sample = (
        #             scale / ((self.max_num_models - 1) ** 0.5)
        #         ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

        #         if fullrank:
        #             w = mean + scaled_diag_sample + cov_sample
        #         else:
        #             w = mean + scaled_diag_sample

        #     else:
        #         w = mean + scaled_diag_sample

        #     module.__setattr__(name, w)

        for var in self.base.trainable_variables:
            name = var.name 
            if var is None:
                continue
            mean = self.params["%s_mean" %name]
            sq_mean = self.params["%s_sq_mean" %name]
            eps = tf.random.normal(shape=mean.shape)

            # only min value clip
            value = tf.clip_by_value(sq_mean - mean ** 2, self.var_clamp, tf.float32.max)
            scaled_diag_sample = scale * tf.math.sqrt(value) * eps

            if cov is True:
                cov_mat_sqrt = self.params["%s_cov_mat_sqrt" % name]
                eps = tf.random.normal(shape=(cov_mat_sqrt.shape[0], 1))
                cov_sample = (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * tf.reshape(tf.linalg.matmul(cov_mat_sqrt, eps, transposed_a=True), shape=mean.shape)

                if fullrank:
                    w = mean + scaled_diag_sample + cov_sample
                else:
                    w = mean + scaled_diag_sample
            else:
                w = mean + scaled_diag_sample
            #self.params[name] = w
            var.assign(w)

    def sample_fullrank(self, scale, cov, fullrank):
        scale_sqrt = scale ** 0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        # for (module, name) in self.params:
        #     mean = module.__getattr__("%s_mean" % name)
        #     sq_mean = module.__getattr__("%s_sq_mean" % name)

        #     if cov:
        #         cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
        #         cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

        #     mean_list.append(mean.cpu())
        #     sq_mean_list.append(sq_mean.cpu())

        for var in self.base.trainable_variables:
            name = var.name
            mean = self.params["%s_mean" % name]
            sq_mean = self.params["%s_sq_mean" %name]

            if cov:
                cov_mat_sqrt = self.params["%s_cov_mat_sqrt" % name]
                cov_mat_sqrt_list.append(cov_mat_sqrt)
            mean_list.append(mean)
            sq_mean_list.append(sq_mean)

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        # var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        # var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        val = tf.clip_by_value(sq_mean - mean ** 2, self.var_clamp, tf.float32.max)
        val_sample = tf.math.sqrt(val) * tf.random.normal(shape=val.shape)

        # if covariance draw low rank sample
        if cov:
            # cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            # cov_sample = cov_mat_sqrt.t().matmul(
            #     cov_mat_sqrt.new_empty(
            #         (cov_mat_sqrt.size(0),), requires_grad=False
            #     ).normal_()
            # )
            # cov_sample /= (self.max_num_models - 1) ** 0.5

            # rand_sample = var_sample + cov_sample

            cov_mat_sqrt = tf.concat(cov_mat_sqrt_list, axis=1)
            cov_sample = tf.linalg.matmul(cov_mat_sqrt, tf.random.noraml(shape=(cov_mat_sqrt.shape[0],1)))
            cov_sample /= (self.max_num_models - 1) ** 0.5
            rand_sample = val_sample + cov_sample

        else:
            #rand_sample = var_sample
            rand_sample = val_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        #sample = sample.unsqueeze(0)
        sample = tf.expand_dims(sample, axis=0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        # for (module, name), sample in zip(self.params, samples_list):
        #     module.__setattr__(name, sample.cuda())
        for var, sample in zip(self.base.trainable_variables, samples_list):
            #self.params[var.name] = sample
            var.assign(sample)

    def collect_model(self, base_model):
        # for (module, name), base_param in zip(self.params, base_model.parameters()):
        #     mean = module.__getattr__("%s_mean" % name)
        #     sq_mean = module.__getattr__("%s_sq_mean" % name)

        #     # first moment
        #     mean = mean * self.n_models.item() / (
        #         self.n_models.item() + 1.0
        #     ) + base_param.data / (self.n_models.item() + 1.0)

        #     # second moment
        #     sq_mean = sq_mean * self.n_models.item() / (
        #         self.n_models.item() + 1.0
        #     ) + base_param.data ** 2 / (self.n_models.item() + 1.0)

        #     # square root of covariance matrix
        #     if self.no_cov_mat is False:
        #         cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

        #         # block covariance matrices, store deviation from current mean
        #         dev = (base_param.data - mean).view(-1, 1)
        #         cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

        #         # remove first column if we have stored too many models
        #         if (self.n_models.item() + 1) > self.max_num_models:
        #             cov_mat_sqrt = cov_mat_sqrt[1:, :]
        #         module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

        #     module.__setattr__("%s_mean" % name, mean)
        #     module.__setattr__("%s_sq_mean" % name, sq_mean)
        
        for var, base_var in zip(self.base.trainable_variables, base_model.trainable_variables):
            name = var.name
            mean = self.params["%s_mean" % name]
            sq_mean = self.params["%s_sq_mean" % name]

            #first moment
            num_models = self.n_models.numpy().item()
            mean = mean * num_models / (num_models + 1.0) + base_var.numpy() / (num_models + 1.0)

            #second moment
            sq_mean = sq_mean * num_models / (num_models + 1.0) + base_var.numpy()**2 / (num_models + 1.0)

            # sqare root of convariance matrix
            if self.no_cov_mat is False:
                cov_mat_sqrt = self.params["%s_cov_mat_sqrt"% name]

                # block covariance matrices, store deviation from current mean
                dev = tf.reshape(base_var.numpy()-mean, shape=(-1,1))
                cov_mat_sqrt = tf.concat([cov_mat_sqrt, tf.transpose(tf.reshape(dev, shape=(-1,1)))], axis=0)

                # remove first column if we have store too many models
                if num_models + 1 > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                self.params["%s_cov_mat_sqrt" % name] = cov_mat_sqrt 
            self.params["%s_mean"%name] = mean 
            self.params["%s_sq_mean"%name] = sq_mean

        #self.n_models.add_(1)
        self.n_models += 1

    # skip in this time
    def load_state_dict(self, state_dict, strict=True):
        # if not self.no_cov_mat:
        #     n_models = state_dict["n_models"].item()
        #     rank = min(n_models, self.max_num_models)
        #     for module, name in self.params:
        #         mean = module.__getattr__("%s_mean" % name)
        #         module.__setattr__(
        #             "%s_cov_mat_sqrt" % name,
        #             mean.new_empty((rank, mean.numel())).zero_(),
        #         )
        # super(SWAG, self).load_state_dict(state_dict, strict)
        raise NotImplementedError

    def export_numpy_params(self, export_cov_mat=False):
        mean_list = []
        sq_mean_list = []
        cov_mat_list = []

        # for module, name in self.params:
        #     mean_list.append(module.__getattr__("%s_mean" % name).cpu().numpy().ravel())
        #     sq_mean_list.append(
        #         module.__getattr__("%s_sq_mean" % name).cpu().numpy().ravel()
        #     )
        #     if export_cov_mat:
        #         cov_mat_list.append(
        #             module.__getattr__("%s_cov_mat_sqrt" % name).cpu().numpy().ravel()
        #         )
        # mean = np.concatenate(mean_list)
        # sq_mean = np.concatenate(sq_mean_list)
        # var = sq_mean - np.square(mean)

        for var in self.base.trainable_variables:
            name = var.name
            mean_list.append(self.params["%s_mean"%name].numpy())
            sq_mean_list.append(self.params["%s_sq_mean" % name].numpy())

            if export_cov_mat:
                cov_mat_list.append(self.params["%s_cov_mat_sqrt" % name].numpy())

        mean = np.concatenate(mean_list)
        sq_mean = np.concatenate(sq_mean_list)
        var = sq_mean - np.square(mean)

        if export_cov_mat:
            return mean, var, cov_mat_list
        else:
            return mean, var

    def import_numpy_weights(self, w):
        k = 0
        for var in self.base.trainable_variables:
            name = var.name
            #mean = module.__getattr__("%s_mean" % name)
            mean = self.params["%s_mean"% name]
            s = np.prod(mean.shape)
            #module.__setattr__(name, mean.new_tensor(w[k : k + s].reshape(mean.shape)))
            #var.assign(tf.reshape(w[k:k+s], shape=mean.shape))
            self.params[name] = tf.reshape(w[k:k+s], shape=mean.shape)
            k += s

    def generate_mean_var_covar(self):
        mean_list = []
        var_list = []
        cov_mat_root_list = []
        for name in self.names:
            # mean = module.__getattr__("%s_mean" % name)
            # sq_mean = module.__getattr__("%s_sq_mean" % name)
            # cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

            mean = self.params["%s_mean"%name]
            sq_mean = self.params["%s_sq_mean" % name]
            cov_mat_sqrt = self.params["%s_cov_mat_sqrt" % name]

            mean_list.append(mean)
            var_list.append(sq_mean - mean ** 2.0)
            cov_mat_root_list.append(cov_mat_sqrt)
        return mean_list, var_list, cov_mat_root_list

    def compute_ll_for_block(self, vec, mean, var, cov_mat_root):
        vec = flatten(vec)
        mean = flatten(mean)
        var = flatten(var)

        # cov_mat_lt = RootLazyTensor(cov_mat_root.t())
        # var_lt = DiagLazyTensor(var + 1e-6)
        # covar_lt = AddedDiagLazyTensor(var_lt, cov_mat_lt)
        # qdist = MultivariateNormal(mean, covar_lt)

        # with gpytorch.settings.num_trace_samples(
        #     1
        # ) and gpytorch.settings.max_cg_iterations(25):
        #     return qdist.log_prob(vec)

        covar = tf.linalg.matmul(cov_mat_root, cov_mat_root, transposed_a=True) + tf.squeeze(tf.linalg.diag(var+1e-6))
        qdist = MultivariateNormal(loc=mean, scale_diag=covar)
        return qdist.log_prob(vec)

    def block_logdet(self, var, cov_mat_root):
        var = flatten(var)

        # cov_mat_lt = RootLazyTensor(cov_mat_root.t())
        # var_lt = DiagLazyTensor(var + 1e-6)
        # covar_lt = AddedDiagLazyTensor(var_lt, cov_mat_lt)

        cov_mat = tf.linalg.matmul(cov_mat_root, cov_mat_root, transposed_a=True)
        var = tf.squeeze(tf.linalg.diag(var+1e-6))
        covar = cov_mat + var 

        return tf.linalg.logdet(covar)

    def block_logll(self, param_list, mean_list, var_list, cov_mat_root_list):
        full_logprob = 0
        for i, (param, mean, var, cov_mat_root) in enumerate(
            zip(param_list, mean_list, var_list, cov_mat_root_list)
        ):
            # print('Block: ', i)
            block_ll = self.compute_ll_for_block(param, mean, var, cov_mat_root)
            full_logprob += block_ll

        return full_logprob

    def full_logll(self, param_list, mean_list, var_list, cov_mat_root_list):
        #cov_mat_root = torch.cat(cov_mat_root_list, dim=1)
        cov_mat_root = tf.concat(cov_mat_root_list, axis=1)
        mean_vector = flatten(mean_list)
        var_vector = flatten(var_list)
        param_vector = flatten(param_list)
        return self.compute_ll_for_block(
            param_vector, mean_vector, var_vector, cov_mat_root
        )

    def compute_logdet(self, block=False):
        _, var_list, covar_mat_root_list = self.generate_mean_var_covar()

        if block:
            full_logdet = 0
            for (var, cov_mat_root) in zip(var_list, covar_mat_root_list):
                block_logdet = self.block_logdet(var, cov_mat_root)
                full_logdet += block_logdet
        else:
            var_vector = flatten(var_list)
            #cov_mat_root = torch.cat(covar_mat_root_list, dim=1)
            cov_mat_root = tf.concat(covar_mat_root_list, axis=1)
            full_logdet = self.block_logdet(var_vector, cov_mat_root)

        return full_logdet

    def diag_logll(self, param_list, mean_list, var_list):
        logprob = 0.0
        for param, mean, scale in zip(param_list, mean_list, var_list):
            #logprob += Normal(mean, scale).log_prob(param).sum()
            logprob += tf.math.reduce_sum(tfp.distributions.Normal(mean, scale).log_prob(param))
        return logprob

    def compute_logprob(self, vec=None, block=False, diag=False):
        mean_list, var_list, covar_mat_root_list = self.generate_mean_var_covar()

        if vec is None:
            #param_list = [getattr(param, name) for param, name in self.params]
            param_list = [param for param in self.base.trainable_variables]
        else:
            param_list = unflatten_like(vec, mean_list)

        if diag:
            return self.diag_logll(param_list, mean_list, var_list)
        elif block is True:
            return self.block_logll(
                param_list, mean_list, var_list, covar_mat_root_list
            )
        else:
            return self.full_logll(param_list, mean_list, var_list, covar_mat_root_list)
