# pylint: disable=too-many-arguments, too-many-locals, too-many-instance-attributes
"""
`ParallLossModule` is a container module that chains a number of modules for part of fc weight, it plays as a role of layer for model parallel on multi context
@author: edison huang 
"""

import logging
import copy
import math

import mxnet as mx
from mxnet.initializer import Uniform

from mxnet.module.base_module import BaseModule

def parall_loss(device_preds, device_labels, ctx, batch_size):
    with mx.autograd.record():
        max_list = [mx.nd.max(pred, axis=1, keepdims=True).copyto(ctx)
                    for pred in device_preds]
        maxmium = mx.nd.max(mx.nd.concat(*max_list, dim=1), axis=1, keepdims=True)
        z_list = [pred - maxmium.copyto(pred.context) for pred in device_preds]
        sum_list = [mx.nd.sum(mx.nd.exp(z), axis=1, keepdims=True).copyto(ctx) for z in z_list]
        log_sum = mx.nd.log(mx.nd.add_n(*sum_list))

        log_softmax_list = [z - log_sum.copyto(z.context) for z in z_list]

        loss_list = [mx.nd.sum(-log_softmax * label) 
                for log_softmax, label in zip(log_softmax_list, device_labels)]
        total_loss = mx.nd.add_n(*[loss.copyto(ctx) for loss in loss_list])/batch_size
        return total_loss

class ParallLossModule(BaseModule):
    """A SequentialModule is a container module that can chain multiple modules together.

    .. note::

        Building a computation graph with this kind of imperative container is less
        flexible and less efficient than the symbolic graph. So, this should be only used as a
        handy utility.
    """

    META_TAKE_LABELS = 'take_labels'
    META_AUTO_WIRING = 'auto_wiring'

    def __init__(self, context, batch_size, ctx_num_class, 
                 local_class_start, logger=logging):
        super(ParallLossModule, self).__init__(logger=logger)
        self._modules = []
        self._sub_module_contexts = []
        self._context = context
        self._batch_size = batch_size
        self._ctx_num_class = ctx_num_class
        self.local_class_start = local_class_start
        assert local_class_start == 0

        self._label_shapes = None
        self._data_shapes = None

    def add(self, module, context, **kwargs):
        """Add a module to the the container

        Parameters
        ----------
        module : BaseModule
            The new module to add.
        context : Context
            The context the module is run in
        kwargs : ``**keywords``
            All the keyword arguments are saved as meta information
            for the added module. The currently known meta includes

            - `take_labels`: indicating whether the module expect to
                take labels when doing computation. Note any module in
                the chain can take labels (not necessarily only the top
                most one), and they all take the same labels passed
                from the original data batch for the `SequentialModule`.


        Returns
        -------
        self
            This function returns `self` to allow us to easily chain a
            series of `add` calls.
        Examples
        --------
        >>> # An example of addinging two modules to a chain.
        >>> seq_mod = mx.mod.ParallLossModule()
        >>> seq_mod.add(part_mod1, ctx1)
        >>> seq_mod.add(part_mod2, ctx2)

        """
        self._modules.append(module)
        self._sub_module_contexts.append(context)
        
        # after adding new modules, we are reset back to raw states, needs
        # to bind, init_params, etc.
        self.binded = False
        self.params_initialized = False
        self.optimizer_initialized = False

        return self # for easier chaining

    @property
    def data_names(self):
        """A list of names for data required by this module."""
        assert len(self._modules) > 0
        return self._modules[0].data_names

    @property
    def output_names(self):
        """A list of names for the outputs of this module."""
        assert False
        if len(self._modules) > 0:
            return self._modules[-1].output_names
        return []

    @property
    def data_shapes(self):
        """Gets data shapes.

        Returns
        -------
        list
            A list of `(name, shape)` pairs. The data shapes of the first module
            is the data shape of a `SequentialModule`.
        """
        assert self.binded
        return self._modules[0].data_shapes

    @property
    def label_shapes(self):
        """Gets label shapes.

        Returns
        -------
        list
            A list of `(name, shape)` pairs. The return value could be `None` if
            the module does not need labels, or if the module is not bound for
            training (in this case, label information is not available).
        """
        assert self.binded
        return self._label_shapes

    @property
    def output_shapes(self):
        """Gets output shapes.

        Returns
        -------
        list
            A list of `(name, shape)` pairs. The output shapes of the last
            module is the output shape of a `SequentialModule`.
        """
        assert self.binded
        return [(name, (shape[0], shape[1] * len(self._modules)))
                for name, shape in self._modules[0].output_shapes]

    def get_params(self):
        """Gets current parameters.

        Returns
        -------
        (arg_params, aux_params)
            A pair of dictionaries each mapping parameter names to NDArray values. This
            is a merged dictionary of all the parameters in the modules.
        """
        assert self.binded and self.params_initialized

        arg_params = dict()
        aux_params = dict()

        for module in self._modules:
            arg, aux = module.get_params()
            arg_params.update(arg)
            aux_params.update(aux)

        return (arg_params, aux_params)

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        """Initializes parameters.

        Parameters
        ----------
        initializer : Initializer
        arg_params : dict
            Default ``None``. Existing parameters. This has higher priority
            than `initializer`.
        aux_params : dict
            Default ``None``. Existing auxiliary states. This has higher priority
            than `initializer`.
        allow_missing : bool
            Allow missing values in `arg_params` and `aux_params` (if not ``None``).
            In this case, missing values will be filled with `initializer`.
        force_init : bool
            Default ``False``.
        allow_extra : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.
        """
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'

        for module in self._modules:
            module.init_params(initializer=initializer, \
                    arg_params=arg_params, aux_params=aux_params, \
                    allow_missing=allow_missing, force_init=force_init,\
                    allow_extra=allow_extra)

        # make sure we do not have duplicated parameter names
        def _check_name(known_names, new_names, modules, i):
            """Internal function to help checking duplicated names."""
            for name in new_names:
                assert not name in known_names, "Duplicated parameter names: " + \
                    ('name "%s" in layer %d (%s) is already ' % (name, i, type(modules[i]))) + \
                    ('used in layer %d (%s).' % (known_names[name],
                                                 type(modules[known_names[name]])))
                known_names[name] = i

        arg_names = dict()
        aux_names = dict()
        for i_layer, module in enumerate(self._modules):
            arg_params, aux_params = module.get_params()
            _check_name(arg_names, arg_params.keys(), self._modules, i_layer)
            _check_name(aux_names, aux_params.keys(), self._modules, i_layer)

        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, 
             shared_module=None, grad_req='write'):
        """Binds the symbols to construct executors. This is necessary before one
        can perform computation with the module.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is `data_iter.provide_data`.
        label_shapes : list of (str, tuple)
            Typically is `data_iter.provide_label`.
        for_training : bool
            Default is ``True``. Whether the executors should be bind for training.
        inputs_need_grad : bool
            Default is ``False``. Whether the gradients to the input data need to be computed.
            Typically this is not needed. But this might be needed when implementing composition
            of modules.
        force_rebind : bool
            Default is ``False``. This function does nothing if the executors are already
            bound. But with this ``True``, the executors will be forced to rebind.
        shared_module : Module
            Default is ``None``. Currently shared module is not supported for `SequentialModule`.
        grad_req : str, list of str, dict of str to str
            Requirement for gradient accumulation. Can be 'write', 'add', or 'null'
            (default to 'write').
            Can be specified globally (str) or for each argument (list, dict).
        """
        if self.binded and not force_rebind:
            self.logger.warning('Already bound, ignoring bind()')
            return

        if inputs_need_grad:
            assert for_training is True
        self.inputs_need_grad = inputs_need_grad
        assert shared_module is None, 'Shared module is not supported'
        assert len(self._modules) > 0, 'Attempting to bind an empty SequentialModule'

        self.binded = True

        # the same label shapes are used for all chained modules
        self._label_shapes = label_shapes


        my_data_shapes = data_shapes
        my_label_shapes = label_shapes
        for i, module in enumerate(self._modules):
            my_inputs_need_grad = True if for_training else False 

            module.bind(data_shapes=my_data_shapes, 
                        label_shapes=my_label_shapes,
                        for_training=for_training, 
                        inputs_need_grad=my_inputs_need_grad,
                        force_rebind=force_rebind,
                        shared_module=None, grad_req=grad_req)
        self._label_shapes = None



    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),),
                       force_init=False):
        """Installs and initializes optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default ``(('learning_rate', 0.01),)``. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default ``False``, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        for module in self._modules:
            module.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                                  optimizer_params=optimizer_params, 
                                  force_init=force_init)

        self.optimizer_initialized = True

    def forward(self, data_batch, is_train=None):
        """Forward computation.

        Parameters
        ----------
        data_batch : DataBatch
        is_train : bool
            Default is ``None``, in which case `is_train` is take as ``self.for_training``.
        """
        assert self.binded and self.params_initialized

        # make a shallow copy, just to maintain necessary properties (if any) like
        # bucket_key, pad, etc.
        data_batch = copy.copy(data_batch)

        for i, module in enumerate(self._modules):
            module.forward(data_batch, is_train=is_train)

        for i, module in enumerate(self._modules):
            assert len(module.get_outputs()) == 1
            module.get_outputs()[0].attach_grad()


        if is_train:
            assert (len(data_batch.label) == 1)
            labels = mx.nd.array(data_batch.label[0], ctx=self._context,
                    dtype='float32') - self.local_class_start
            #labels = mx.nd.split(
            #        mx.nd.one_hot(mx.nd.array(data_batch.label[0], 
            #            ctx=self._context, dtype='float32'), self._num_class),
            #        num_outputs=len(self._modules), axis=1)
            device_labels = []
            for i, ctx in enumerate(self._sub_module_contexts):
                device_labels.append(
                        mx.nd.one_hot(labels.as_in_context(ctx) - (i*self._ctx_num_class), 
                                    self._ctx_num_class))
            total_loss = parall_loss([mod.get_outputs()[0] for mod in self._modules], device_labels, 
                            self._context, self._batch_size)
            assert not math.isnan(total_loss.asscalar())
            total_loss.backward()
            
    def backward(self, out_grads=None):
        """Backward computation."""
        assert self.binded and self.params_initialized
        assert out_grads == None
        assert len(self._modules) > 1

        for  i, module in enumerate(self._modules):
            module.backward(out_grads=[self._modules[i].get_outputs()[0].grad])

    def update(self):
        """Updates parameters according to installed optimizer and the gradient computed
        in the previous forward-backward cycle.
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized

        for module in self._modules:
            module.update()

    def get_outputs(self, merge_multi_context=True):
        """Gets outputs from a previous forward computation.

        Parameters
        ----------
        merge_multi_context : bool
            Default is ``True``. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A ``True`` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        list of NDArray or list of list of NDArray
            If `merge_multi_context` is ``True``, it is like ``[out1,
            out2]``. Otherwise, it is like ``[[out1_dev1, out1_dev2], [out2_dev1,
            out2_dev2]]``. All the output elements are numpy arrays.
        """
        assert False ## will have no change to get output of the class layer
        assert self.binded and self.params_initialized
        pred_list = [m.get_outputs(merge_multi_context=merge_multi_context) 
                        for m in self._modules]

        num_outputs = len(pred_list[0])
        result = []
        for i in range(num_outputs):
            out = [outputs[i] for outpus in pred_list]
            tensor = mx.nd.concat(*[temp.as_in_context(self._context) for temp in out], dim=1)
            result.append(tensor)
        return result

    def get_input_grads(self, merge_multi_context=True):
        """Gets the gradients with respect to the inputs of the module.

        Parameters
        ----------
        merge_multi_context : bool
            Default is ``True``. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A ``True`` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        list of NDArrays or list of list of NDArrays
            If `merge_multi_context` is ``True``, it is like ``[grad1, grad2]``. Otherwise, it
            is like ``[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]``. All the output
            elements are `NDArray`.
        """

        assert self.binded and self.params_initialized and self.inputs_need_grad
        grad_list = [m.get_input_grads(merge_multi_context=merge_multi_context)
                for m in self._modules]
        mod_len = len(self._modules)
        grad_len = len(grad_list[0])
        return [mx.nd.add_n(*[grad_list[m][g].as_in_context(self._context) for m in range(mod_len)])
                for g in range(grad_len)]

    def update_metric(self, eval_metric, labels, pre_sliced=False):
        """Evaluates and accumulates evaluation metric on outputs of the last forward computation.

        Parameters
        ----------
        eval_metric : EvalMetric
        labels : list of NDArray
            Typically ``data_batch.label``.
        """
        assert self.binded and self.params_initialized
       
        def parall_argmax(datas, ctx):
            sub_max = mx.nd.concat(*[mx.nd.max(data, axis=1, keepdims=True).as_in_context(self._context)
                                        for data in datas])
            sub_arg_max = mx.nd.concat(*[data.shape[1]* i + mx.nd.argmax(data, axis=1, keepdims=True).as_in_context(self._context)
                                        for i, data in enumerate(datas)], dim=1)
            part_arg_max = mx.nd.argmax(sub_max, axis=1)
            return mx.nd.pick(sub_arg_max, part_arg_max)

        preds = parall_argmax([mod.get_outputs()[0] for mod in self._modules], self._context)
        assert (len(labels) == 1)
        eval_metric.update_dict({'label':mx.nd.array(labels[0])}, {'fc_logits':preds})
        

    def install_monitor(self, mon):
        """Installs monitor on all executors."""
        assert self.binded
        for module in self._modules:
            module.install_monitor(mon)
