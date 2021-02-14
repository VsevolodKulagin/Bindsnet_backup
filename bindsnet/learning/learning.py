from abc import ABC
from typing import Union, Optional, Sequence

import torch
import numpy as np

from ..network.nodes import SRM0Nodes
from ..network.topology import (
    AbstractConnection,
    Connection,
    Conv2dConnection,
    LocalConnection,
)
from ..utils import im2col_indices


class LearningRule(ABC):
    # language=rst
    """
    Abstract base class for learning rules.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        # Connection parameters.
        self.connection = connection
        self.source = connection.source
        self.target = connection.target

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        # Learning rate(s).
        if nu is None:
            nu = [0.0, 0.0]
        elif isinstance(nu, float) or isinstance(nu, int):
            nu = [nu, nu]

        self.nu = nu

        # Parameter update reduction across minibatch dimension.
        if reduction is None:
            reduction = torch.mean

        self.reduction = reduction

        # Weight decay.
        self.weight_decay = weight_decay

    def update(self) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        # Implement weight decay.
        if self.weight_decay:
            self.connection.w -= self.weight_decay * self.connection.w

        # Bound weights.
        if (
            self.connection.wmin != -np.inf or self.connection.wmax != np.inf
        ) and not isinstance(self, NoOp):
            self.connection.w.clamp_(self.connection.wmin, self.connection.wmax)


class NoOp(LearningRule):
    # language=rst
    """
    Learning rule with no effect.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object which this learning rule will have no effect on.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        super().update()


class PostPre(LearningRule):
    # language=rst
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity. The pre-synaptic update is negative, while
    the post-synpatic update is positive.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``PostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection`` class.
        """
        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

        # Pre-synaptic update.
        if self.nu[0]:
            update = self.reduction(torch.bmm(source_s, target_x), dim=0)
            self.connection.w -= self.nu[0] * update

        # Post-synaptic update.
        if self.nu[1]:
            update = self.reduction(torch.bmm(source_x, target_s), dim=0)
            self.connection.w += self.nu[1] * update

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        if self.nu[0]:
            pre = self.reduction(
                torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
            )
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1]:
            post = self.reduction(
                torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
            )
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()


class WeightDependentPostPre(LearningRule):
    # language=rst
    """
    STDP rule involving both pre- and post-synaptic spiking activity. The post-synaptic update is positive and the pre-
    synaptic update is negative, and both are dependent on the magnitude of the synaptic weights.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        tc_trace: float = 20,
        tc_trace_neg: float = 20,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``WeightDependentPostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``WeightDependentPostPre`` learning rule will
                           modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )
        
        self.tc_trace = tc_trace
        self.tc_trace_neg = tc_trace_neg
        
        self.interval = 100
        
        self.fl = open("C:/Users/Всеволод/Desktop/STDP.txt",'r')
        
        self.STDP_base =  torch.ones([101, 120])
        
        i = 0
        k = 0
        for line in self.fl:
    
            for sym in line.split():
                
                self.STDP_base[i][k]= float(sym) 
                k += 1
            k = 0            
            i += 1
        i = 0       
        self.fl.close()
        
        
        assert self.source.traces, "Pre-synaptic nodes must record spike traces."
        assert (
            connection.wmin != -np.inf and connection.wmax != np.inf
        ), "Connection must define finite wmin and wmax."

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )
            
    def G_neg(self, R, g_neg= 0.137, G_0_neg= -3.5, R_G_neg= 1230):       
        return G_0_neg*(g_neg + np.exp(-R/R_G_neg))
    
    def G_pos(self, R, g_pos= 23.2, G_0_pos= 0.046, R_G_pos= 790):
        return G_0_pos*(g_pos + R*np.exp(-R/R_G_pos))
            
    def c_neg(self, R, c_c_neg= -0.53, a_0_neg= 1.0, R_c_neg= 810):
        #return a_0_neg*(c_c_neg + np.exp(-R/R_c_neg))
        return 0
    
    def c_pos(self, R, c_c_pos= 0.01, a_0_pos= -1.7, R_c_pos= 130):
        #return a_0_pos*(c_c_pos + np.exp(-R/R_c_pos))    
        return 0
        
    def Ohm_to_weight(self, Ohm, Ohm_min=1000, Ohm_max=10000):
        weight = 1/(Ohm)
        return (weight - 1/Ohm_max) / (1/Ohm_min - 1/Ohm_max)
    
    def weight_to_Ohm(self, weight, Ohm_min=1000, Ohm_max=10000):
        Ohm = 1 / (weight * (1/Ohm_min - 1/Ohm_max) + 1/Ohm_max)
        return Ohm
    
    def round_to_percent(self, tensor):
        
        tensor = tensor *100
        
        #print(float(tensor[1,1]))
        
        for i in range (tensor.size(0)):
            
            for k in range (tensor.size(1)):
            
                tensor[i][k] = int(round(float(tensor[i][k])))
        
        return tensor
        
    
    def delta_w_custom(self, delta):
        
        first_index = self.round_to_percent(self.connection.w)
        second_index = (delta + 60)
        for i in range (first_index.size(0)):
            
            for k in range (first_index.size(1)):
                
                if (second_index[i][k] > 120) or (second_index[i][k] < 0):
                
                    second_index[i][k] = 0
                
                delta[i][k] = self.STDP_base[int(first_index[i][k])][int(second_index[i][k])]
                
               # if (second_index[i][k] > 120) or (second_index[i][k] < 0):
                    
                 #   delta[i][k] = 0
               # if  delta[i][k]<0:
                    
                    #print(self.source.s.view(self.source.batch_size, -1).unsqueeze(2).float(),'\n')
                    
               
        return delta
    
    
    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection`` class.
        """
        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x_neg.view(batch_size, -1).unsqueeze(1)

        update = 0
        Ohm_max = 10000
        Ohm_min = 1000

        G_neg = self.G_neg(self.weight_to_Ohm(self.connection.w, Ohm_min, Ohm_max))
        G_pos = self.G_pos(self.weight_to_Ohm(self.connection.w, Ohm_min, Ohm_max))
        c_neg = self.c_neg(self.weight_to_Ohm(self.connection.w, Ohm_min, Ohm_max))
        c_pos = self.c_pos(self.weight_to_Ohm(self.connection.w, Ohm_min, Ohm_max))
        
        #print(G_neg.size(), G_pos.size(), c_neg.size(),c_pos.size())
        #for i in range (source_s.size(1)):
          #  c_neg[i]= c_neg[i]*source_s[0,i]
        
       # for i in range (target_s.size(2)):
        #    for k in range (c_pos.size(0)):
         #       c_pos[k,i]= c_pos[k,i]*target_s[0,0,i]
        
        # Pre-synaptic update.
        
        
        outer_product = self.reduction(torch.bmm(source_s, target_x), dim=0)
        update += self.nu[0] * self.delta_w_custom(self.tc_trace_neg*np.log(outer_product)) # + c_neg)
       # print(-self.tc_trace_neg*np.log(outer_product))
        
        # Post-synaptic update.
        
        outer_product = self.reduction(torch.bmm(source_x, target_s), dim=0)
        update += self.nu[1] * self.delta_w_custom(-self.tc_trace*np.log(outer_product)) # + c_pos)
       # update += update_temp
        
        self.connection.w += update

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        (
            out_channels,
            in_channels,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        update = 0

        # Pre-synaptic update.
        if self.nu[0]:
            pre = self.reduction(
                torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
            )
            update -= (
                self.nu[0]
                * pre.view(self.connection.w.size())
                * (self.connection.w - self.wmin)
            )

        # Post-synaptic update.
        if self.nu[1]:
            post = self.reduction(
                torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
            )
            update += (
                self.nu[1]
                * post.view(self.connection.w.size())
                * (self.wmax - self.connection.wmin)
            )

        self.connection.w += update

        super().update()


class Hebbian(LearningRule):
    # language=rst
    """
    Simple Hebbian learning rule. Pre- and post-synaptic updates are both positive.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``Hebbian`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``Hebbian`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Connection`` subclass of ``AbstractConnection`` class.
        """
        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

        # Pre-synaptic update.
        update = self.reduction(torch.bmm(source_s, target_x), dim=0)
        self.connection.w += self.nu[0] * update

        # Post-synaptic update.
        update = self.reduction(torch.bmm(source_x, target_s), dim=0)
        self.connection.w += self.nu[1] * update

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.
        """
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        pre = self.reduction(torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0)
        self.connection.w += self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = self.reduction(torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0)
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()


class MSTDP(LearningRule):
    # language=rst
    """
    Reward-modulated STDP. Adapted from `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``MSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDP`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param tc_plus: Time constant for pre-synaptic firing trace.
        :param tc_minus: Time constant for post-synaptic firing trace.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(batch_size, *self.source.shape)
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(batch_size, *self.target.shape)
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(batch_size, *self.connection.w.shape)

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(batch_size, -1).float()
        target_s = self.target.s.view(batch_size, -1).float()

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        # Compute weight update based on the point eligibility value of the past timestep.
        update = reward * self.eligibility
        self.connection.w += self.nu[0] * self.reduction(update, dim=0)

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(
            self.p_plus.unsqueeze(2), target_s.unsqueeze(1)
        ) + torch.bmm(source_s.unsqueeze(2), self.p_minus.unsqueeze(1))

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(batch_size, *self.connection.w.shape)

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        batch_size = self.source.batch_size

        # Compute weight update based on the point eligibility value of the past timestep.
        update = reward * self.eligibility
        self.connection.w += self.nu[0] * torch.sum(update, dim=0)

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(batch_size, *self.source.shape)
            self.p_plus = im2col_indices(
                self.p_plus, kernel_height, kernel_width, padding=padding, stride=stride
            )
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(batch_size, *self.target.shape)
            self.p_minus = self.p_minus.view(batch_size, out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(
            target_s, self.p_plus.permute((0, 2, 1))
        ) + torch.bmm(self.p_minus, source_s.permute((0, 2, 1)))
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()


class MSTDPET(LearningRule):
    # language=rst
    """
    Reward-modulated STDP with eligibility trace. Adapted from
    `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``MSTDPET`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDPET`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param float tc_plus: Time constant for pre-synaptic firing trace.
        :param float tc_minus: Time constant for post-synaptic firing trace.
        :param float tc_e_trace: Time constant for the eligibility trace.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 25.0))

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        # Initialize eligibility, eligibility trace, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(self.source.n)
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(self.target.n)
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(*self.connection.w.shape)
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(*self.connection.w.shape)

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(-1).float()
        target_s = self.target.s.view(-1).float()

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        # Calculate value of eligibility trace based on the value of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)
        self.eligibility_trace += self.eligibility / self.tc_e_trace

        # Compute weight update.
        self.connection.w += (
            self.nu[0] * self.connection.dt * reward * self.eligibility_trace
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.ger(self.p_plus, target_s) + torch.ger(
            source_s, self.p_minus
        )

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility and eligibility trace.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(batch_size, *self.connection.w.shape)
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(batch_size, *self.connection.w.shape)

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        # Calculate value of eligibility trace based on the value of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)

        # Compute weight update.
        update = reward * self.eligibility_trace
        self.connection.w += self.nu[0] * self.connection.dt * torch.sum(update, dim=0)

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(batch_size, *self.source.shape)
            self.p_plus = im2col_indices(
                self.p_plus, kernel_height, kernel_width, padding=padding, stride=stride
            )
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(batch_size, *self.target.shape)
            self.p_minus = self.p_minus.view(batch_size, out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = (
            self.target.s.permute(1, 2, 3, 0).view(batch_size, out_channels, -1).float()
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(
            target_s, self.p_plus.permute((0, 2, 1))
        ) + torch.bmm(self.p_minus, source_s.permute((0, 2, 1)))
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()


class Rmax(LearningRule):
    # language=rst
    """
    Reward-modulated learning rule derived from reward maximization principles. Adapted from
    `(Vasilaki et al., 2009) <https://intranet.physio.unibe.ch/Publikationen/Dokumente/Vasilaki2009PloSComputBio_1.pdf>`_.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``R-max`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``R-max`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param float tc_c: Time constant for balancing naive Hebbian and policy gradient learning.
        :param float tc_e_trace: Time constant for the eligibility trace.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        # Trace is needed for computing epsilon.
        assert (
            self.source.traces and self.source.traces_additive
        ), "Pre-synaptic nodes should keep track of their firing trace in an additive way."

        # Derivation of R-max depends on stochastic SRM neurons!
        assert isinstance(
            self.target, SRM0Nodes
        ), "R-max needs stochastically firing neurons, use SRM0Nodes."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_c = torch.tensor(
            kwargs.get("tc_c", 5.0)
        )  # 0 for pure naive Hebbian, inf for pure policy gradient.
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 25.0))

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        R-max learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement learning task.
        """
        # Initialize eligibility trace.
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(*self.connection.w.shape)

        # Reshape variables.
        target_s = self.target.s.view(-1).float()
        target_s_prob = self.target.s_prob.view(-1)
        source_x = self.source.x.view(-1)

        # Parse keyword arguments.
        reward = kwargs["reward"]

        # New eligibility trace.
        self.eligibility_trace *= 1 - self.connection.dt / self.tc_e_trace
        self.eligibility_trace += (
            target_s
            - (target_s_prob / (1.0 + self.tc_c / self.connection.dt * target_s_prob))
        ) * source_x[:, None]

        # Compute weight update.
        self.connection.w += self.nu[0] * reward * self.eligibility_trace

        super().update()
