#!/usr/bin/env python3

"""
Filename: reservoir.py

Description:
    A class that simulates the dynamics of a reservoir computer with an RK4
    DDE solver. 

    It is built with JAX, specifically JIT compilation, for speed.

    It should be able to handle any given DDE equation with possibly some
    input changes.
"""

OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1

import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

import os

from PIL import Image, ImageDraw, ImageFont
from IPython.display import HTML, display

from io import BytesIO

class Reservoir:
    def __init__(self, *, h=.02, theta, N, sd=0, amp=1, tau=0, fb_gain=0, 
                 norm_factor=None, norm_offset=None, state_shape=None,
                 input_connectivity=None, mask=None, mask_seed=0,
                 load=False, normalize_mask=True, VDC=20):

        self.h = h
        self.theta = theta
        self.N = N
        self.sd = int(sd)
        self.amplification = amp
        self.tau = tau
        self.fb_gain = fb_gain
        self.mask_seed = mask_seed
        self.VDC = VDC

        self.tau_steps = int(self.tau * jnp.floor(self.theta/self.h))
        m = self.tau_steps
        self.buf_len = max(m+2, 2)
        self.pos_buf = jnp.full((self.buf_len,), 0.0, dtype=jnp.float32)
        self.vel_buf = jnp.full((self.buf_len,), 0.0, dtype=jnp.float32)
        self.buf_idx = jnp.int32(0)
        self.buf_cnt = jnp.int32(0)

        self.stride = int(jnp.maximum(1, jnp.floor(self.theta / self.h)))
        if self.sd > self.stride:
            raise ValueError("Sample delay must be less than or equal to stride")
        self.n_steps_needed = (self.N) * self.stride + 1

        self.normalization_factor = norm_factor
        self.normalization_offset = norm_offset
        if type(state_shape) is not int:
            state_shape = int(state_shape[-1])
            print(f"Note: state_shape was not int, using {state_shape}")
        self.state_shape = state_shape
        # print(state_shape)

        self.input_connectivity = input_connectivity
        
        if load:
            self.mask = mask
        else:
            self.mask = self.create_mask(mask_seed, normalize_mask)



    def sim(self, *, obs=None, Vin=None, full_data=False, direct_fb=False, get_Vin=False):

        if Vin is None:
            state = np.array(obs).reshape(1,self.state_shape)
            u = np.divide(state + self.normalization_offset, self.normalization_factor)
            Vin = np.dot(u, self.mask)
            Vin = Vin.reshape(-1)
            if get_Vin:
                return Vin*self.amplification + self.VDC
            # print(f'Vin mean new: {np.mean(Vin)}')

        positions, theta_vals, use_fb, self.pos_buf, self.vel_buf, self.buf_idx, self.buf_cnt = Reservoir.rk4(
            n_steps=self.n_steps_needed,
            # y0=jnp.asarray(self.MEMS_IC, jnp.float32),
            h=float(self.h),
            Vin=jnp.asarray(Vin, jnp.float32),
            theta=float(self.theta),   # keyword-only & static
            N=int(self.N),             # keyword-only & static
            sd=self.sd,  # keyword-only & static
            amp=float(self.amplification),
            VDC=float(self.VDC),
            stride=self.stride,
            tau=self.tau_steps,
            pos_buf=jnp.asarray(self.pos_buf, jnp.float32),
            vel_buf=jnp.asarray(self.vel_buf, jnp.float32),
            fb_gain=float(self.fb_gain),
            buf_idx=jnp.asarray(self.buf_idx),
            buf_cnt=jnp.asarray(self.buf_cnt)

        )

        if direct_fb:
            # add a bias to neuron readings of u and an additional 1
            # neuron vals + [u] + [1]
            theta_vals = np.array(theta_vals).reshape(1,-1)
            neuron_vals = np.append(theta_vals,np.append(u,[[1]],axis=1), axis=1)
            return neuron_vals

        if full_data:
            return positions, theta_vals, Vin
        else:
            return np.array(theta_vals)

    def create_mask(self, seed, normalize_mask=False):
        mask_rand = np.random.RandomState(seed)
        mask = mask_rand.choice([-1,1],size=(self.state_shape, self.N))

        # add sparsity to mask
        REPLACE_COUNT = np.int32(self.input_connectivity * self.state_shape * self.N)
        mask_reshaped = np.reshape(mask, [self.state_shape * self.N,])
        mask_reshaped[mask_rand.choice(self.state_shape * self.N, REPLACE_COUNT, replace=False)] = 0

        mask = np.reshape(mask_reshaped,[self.state_shape, self.N])

        if normalize_mask:
            mask = mask / self.state_shape

        return mask
    
    def zero_reservoir(self):
        self.pos_buf = jnp.full((self.buf_len,), 0.0, dtype=jnp.float32)
        self.vel_buf = jnp.full((self.buf_len,), 0.0, dtype=jnp.float32)
        self.buf_idx = jnp.int32(0)
        self.buf_cnt = jnp.int32(0)
    
    def get_neuron_sat(self, neurons):
        return np.sum(np.abs(neurons)>=0.89)*100/neurons.size
    
    @staticmethod
    @partial(jax.jit, static_argnames=('n_steps','h','theta','N', 'sd', 'stride', 'tau', 'fb_gain', 'amp', 'VDC'))
    def rk4(*, n_steps, pos_buf, vel_buf, h, Vin, theta, N, sd, amp, stride, tau, fb_gain, buf_idx, buf_cnt, VDC):
        """
        Simulate and sample a DDE for a given set of step input voltages with RK4
        and cubic Hermite interpolation for the delay term. The state of the DDE 
        is stored in ring buffers to allow for efficient and JAX correct sampling 
        and delayed feedback.
        

        Args:
            n_steps (int): total number of simulation steps.
            pos_buf (jnp.ndarray): ring buffer to store past positions, length buf_len.
            vel_buf (jnp.ndarray): ring buffer to store past velocities, length buf_len.
            h (float): step size for the simulation.
            Vin (jnp.ndarray): input voltage array, length n_steps.
            theta (float): sample time for one reservoir reading [non-dimensional].
            N (int): number of reservoir states.
            sd (int): sample delay (in timesteps).
            stride (int): sample every 'stride' with delay 'start'.
            tau (float): delayed feedback gain parameter [non-dimensional].
            fb_gain (float): feedback gain parameter [non-dimensional].
            amp (float): amplitude of the input voltage [non-dimensional].
            buf_idx (int): next write index (most recent sample is at buf_idx-1).
            buf_cnt (int): number of valid samples in buffers (<= len(pos_buf)).
            VDC (float): DC offset voltage [V].

        Returns:
            positions (jnp.ndarray): all positions from simulation.
            sampled (jnp.ndarray): sampled positions (neuron values), length n_samples.
            pbuf_out (jnp.ndarray): updated ring buffer of past positions.
            vbuf_out (jnp.ndarray): updated ring buffer of past velocities.
            buf_idx_out (int): updated next write index.
            buf_count_out (int): updated number of valid samples in buffers.

        """

        # time grid (for indices only)
        tgrid = jnp.arange(n_steps, dtype=jnp.int32)  # 0 ... n_steps-1
        th = tgrid * h

        # precompute Vin indices for k1 (t), k2/k3 (t+h/2), k4 (t+h)
        idx1 = jnp.floor_divide(th, theta).astype(jnp.int32)
        idx2 = jnp.floor_divide(th + 0.5*h, theta).astype(jnp.int32)
        idx4 = jnp.floor_divide(th + h, theta).astype(jnp.int32)

        # clamp indices to valid range
        vmax = jnp.int32(Vin.shape[0] - 1)
        idx1 = jnp.clip(idx1, 0, vmax)
        idx2 = jnp.clip(idx2, 0, vmax)
        idx4 = jnp.clip(idx4, 0, vmax)

        # gather per-step voltages (all on device)
        V1 = Vin[idx1]  # [n_steps]
        V2 = Vin[idx2]
        V4 = Vin[idx4]

        # init state
        buf_len = pos_buf.shape[0]
        last_idx = jnp.mod(buf_idx-1, buf_len)
        y0f = jnp.asarray((pos_buf[last_idx],vel_buf[last_idx]), jnp.float32)
        h32 = jnp.float32(h)
        fb_gain = jnp.float32(fb_gain)
        use_fb = tau > 0
        tau = jnp.float32(tau)

        # in discrete domain where h has been transformed into ints
        # tau is h steps back of delay

        # ring buffers
        m = jnp.floor(tau).astype(jnp.int32)

        init_pos_buf = jnp.asarray(pos_buf, jnp.float32) # jnp.full((buf_len,), y0f[0], dtype=jnp.float32)  # prefill with initial pos
        init_vel_buf = jnp.asarray(vel_buf, jnp.float32)

        # carry.shape = (y, pos_ringbuf, i)
        init_carry = (y0f, init_pos_buf, init_vel_buf, jnp.int32(0), jnp.int32(buf_idx), jnp.int32(buf_cnt))

        def hermite_unit(x0, x1, v0, v1, unit):
            # Cubic Hermite on unit interval, unit in [0,1]
            # unit = (time - tau) - floor(time - tau) roughly
            # H1 = 2*unit^3 - 3*unit^2 + 1
            # H2 = -2*unit^3 + 3*unit^2
            # H3 = unit^3 - 2*unit^2 + unit
            # H4 = unit^3 - unit^2
            # return H1*x0 + H2*x1 + H3*v0 + H4*v1
            t2 = unit * unit
            t3 = t2 * unit
            H1 =  2.0*t3 - 3.0*t2 + 1.0
            H2 = -2.0*t3 + 3.0*t2
            H3 =  t3 - 2.0*t2 + unit
            H4 =  t3 - t2
            return H1*x0 + H2*x1 + H3*v0 + H4*v1

        def sample_delay_hermite_ring(i, c, tau, pos_buf, vel_buf, buf_idx, buf_cnt, buf_len, m):
            """
            Evaluate delayed position x(t_i + c - tau_steps) from a ring buffer using cubic Hermite.
            pos_buf/vel_buf: ring buffers storing past samples at unit step spacing.
            buf_idx: next write index (most recent sample is at buf_idx-1).
            buf_cnt: number of valid samples in buffers (<= len(pos_buf)).
            """
            # s = i + c - tau which is the s = timestep + rk4 offset - tau
            s = (i.astype(jnp.float32) + jnp.float32(c)) - jnp.float32(tau)

            j = jnp.floor(s).astype(jnp.int32)     # left node
            unit = s - jnp.floor(s)               # fractional in [0,1)

            # ex. i=0, c=0,   tau=10, s=-10,  j=-10, unit = -10-(-10)  = 0
            # ex. i=0, c=0.5, tau=10, s=-9.5, j=-9,  unit = -9.5-(-10) = 0.5
            # ex. i=0, c=1,   tau=10, s=-9,   j=-9,  unit = -9-(-9)    = 0

            # ex. i=3, c=0.5, tau=7,  s=-3.5  j=-4,  unit = -3.5-(-4)  = 0.5
            # ex. i=7, c=0.5, tau=7,  s=0.5,  j=0,   unit = 0.5-0      = 0.5

            # Map logical index j to ring-buffer index. Latest logical index is i-1 at (buf_idx-1).
            # So ring index for j is: buf_idx + j - i  (mod buf_len). For j+1: +1.
            j0 = (buf_idx + j - i) % buf_len
            j1 = (buf_idx + j + 1 - i) % buf_len

            x0 = pos_buf[j0]; x1 = pos_buf[j1]
            v0 = vel_buf[j0]; v1 = vel_buf[j1]  # velocity = dx/dt

            # We need at least m+1 samples available to safely interpolate,
            # where m = floor(tau_steps). This covers the worst case at c=0, i=0.

            have_hist = buf_cnt >= (m + 1)

            # Fallback (early warm-up): simple midpoint linear estimate
            x_lin = 0.5 * (x0 + x1)
            # x_h   = hermite_unit(x0, x1, v0, v1, unit)
            x_h   = hermite_unit(x0, x1, v0*h32, v1*h32, unit)
            return jax.lax.select(have_hist, x_h, x_lin)

        def step(carry, Vs):
            y, pbuf, vbuf, i, widx, cnt = carry
            V1_t, V2_t, V4_t = Vs

            if use_fb:
                x_tau_c0  = sample_delay_hermite_ring(i, 0.0, tau, pbuf, vbuf, widx, cnt, buf_len, m)
                x_tau_c05 = sample_delay_hermite_ring(i, 0.5, tau, pbuf, vbuf, widx, cnt, buf_len, m)
                x_tau_c1  = sample_delay_hermite_ring(i, 1.0, tau, pbuf, vbuf, widx, cnt, buf_len, m)

                fb_pos1 = fb_gain * x_tau_c0
                fb_pos2 = fb_gain * x_tau_c05
                fb_pos4 = fb_gain * x_tau_c1
            else:
                # feedback = 0 if tau=0
                fb_pos1 = fb_pos2 = fb_pos4 = jnp.float32(0.0)

            # apply feedback
            V1_eff = V1_t + fb_pos1
            V2_eff = V2_t + fb_pos2
            V4_eff = V4_t + fb_pos4

            k1 = h32 * Reservoir.diff_eq(y, V1_eff, amp, VDC)
            k2 = h32 * Reservoir.diff_eq(y + 0.5 * k1, V2_eff, amp, VDC)
            k3 = h32 * Reservoir.diff_eq(y + 0.5 * k2, V2_eff, amp, VDC)
            k4 = h32 * Reservoir.diff_eq(y + k3, V4_eff, amp, VDC)

            y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            # stopper at pos >= 0.9
            pos, vel = y_next[0], y_next[1]
            clamp = pos >= 0.9
            pos = jnp.where(clamp, jnp.float32(0.9), pos)
            # vel = jnp.where(clamp, jnp.float32(0.0), vel)
            vel = jnp.where(clamp & (vel > 0), jnp.float32(0.0), vel)

            y_next = jnp.array([pos, vel], dtype=jnp.float32)

            pbuf_next = pbuf.at[widx].set(y_next[0])
            vbuf_next = vbuf.at[widx].set(y_next[1])
            widx_next = jnp.mod(widx+1, buf_len)
            cnt_next = jnp.minimum(cnt+1, jnp.int32(buf_len))

            return (y_next, pbuf_next, vbuf_next, i+1, widx_next, cnt_next), y_next

        # run scan
        Vs_all = (V1, V2, V4)                                # tuple of [n_steps] arrays
        (y_last, pbuf_out, vbuf_out, _, buf_idx_out, buf_count_out), Ys = jax.lax.scan(step, init_carry, Vs_all)  # Ys: [n_steps, 2]
        #                            f, init carry,     xs
        # .scan scans (iterates?) a function over an array while carrying along state

        # sample every 'stride' with delay 'start'
        positions = Ys[:, 0]                                  # [n_steps]
        sampled = positions[sd:sd+N*stride:stride]
                            # [n_samples]

        buf_idx = jnp.mod(buf_idx, buf_len)
        return (
            positions,
            sampled,
            use_fb,
            pbuf_out,
            vbuf_out,
            buf_idx_out,
            buf_count_out
        )
    
    @staticmethod
    def diff_eq(y, V, amp, VDC):
        """
        MEMS differential equation.

        Args:
            y (jnp.ndarray): Current state of the MEMS device [position, velocity].
            V (jnp.ndarray): Input voltage at current time [Volts].
            amp (jnp.float): Amplification parameter for the reservoir.
        """
        zeta = jnp.float32(0.2)
        eps  = jnp.float32(8.85e-12)
        k    = jnp.float32(215.0)
        A    = jnp.float32(39.6e-6)
        d    = jnp.float32(42e-6)
        VDC  = jnp.float32(VDC)
        Amp  = jnp.float32(amp)

        pos, vel = y[0], y[1]
        den = 2.0 * k * (d**3) * jnp.maximum(1e-6, (1.0 - pos)**2)
        acc = (eps * A * (Amp * V + VDC)**2) / den - 2.0 * zeta * vel - pos
        return jnp.array([vel, acc], dtype=jnp.float32)

class AnalyzeReservoir:
    def __init__(self, reservoir):
        self.reservoir = reservoir

    def plot_mask(self, *, together=True, separate=True, combined=True, 
                  cmap_color='tab10', title_addition="", save=False, folder_path=None,
                  base_file_name="", figsize=(3.5,3), ax=None, show=True, color='royalblue',
                  partial_N=None):

        if ax is None:
            fig, ax = plt.figure(figsize=figsize, dpi=150)
        else:
            fig = ax.figure

        if folder_path is None and save:
            print('No folder provided, plot will not be saved.')
            save = False
        
        if folder_path is not None:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        

        cmap = plt.get_cmap(cmap_color)
        colors = [cmap(i) for i in range(self.reservoir.state_shape)]
        if together:
            
            for i in range(self.reservoir.state_shape):
                y_axis = np.repeat(self.reservoir.mask[i, :], 50)
                x_axis = np.linspace(0, self.reservoir.N-1, len(y_axis))
                plt.plot(x_axis, y_axis, label=f'Mask {i+1}', color=colors[i])

            
            plt.xlabel('Neuron')
            plt.ylabel('Weight')
            plt.title(f'Mask for Each State {title_addition}') if title_addition != "" else None
            plt.legend()
            plt.grid()
            if save:
                plt.savefig(f'{folder_path}/{base_file_name}_together_masks.png')
        
        if separate:
            
            for i in range(self.reservoir.state_shape):
                plt.figure(figsize=figsize, dpi=150)
                y_axis = np.repeat(self.reservoir.mask[i, :], 50)
                x_axis = np.linspace(0, self.reservoir.N-1, len(y_axis))
                plt.plot(x_axis, y_axis, label=f'Mask {i+1}', color=colors[i])
                plt.plot(self.reservoir.mask[i, :], color='black')
                plt.xlabel('Neuron')
                plt.ylabel('Weight')
                plt.title(f'Mask {i+1} {title_addition}') if title_addition != "" else None
                plt.legend()
                plt.grid()
                if save:
                    plt.savefig(f'{folder_path}/{base_file_name}_mask_{i+1}.png')

        if combined:
            if partial_N is not None:
                mask_partial = self.reservoir.mask[:, :partial_N]
            else:
                partial_N = self.reservoir.N
                mask_partial = self.reservoir.mask
            y_axis = np.repeat(mask_partial.sum(axis=0), 50)
            x_axis = np.linspace(0, partial_N, len(y_axis))
            ax.plot(x_axis, y_axis, label='Combined Mask', color=color, linewidth=1)
            # ax.set_xlabel('Neuron')
            ax.set_ylabel('Mask Weight', labelpad=0)
            ax.set_xticks([0, 10, 20])
            ax.set_xticklabels(['0', '10', '20',], fontsize=8)
            ax.set_ylim(-1.1, 1.1)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xlim(0, partial_N)
            ax.set_title('a) Combined Mask', pad=6, loc='left', fontsize=9)
            # plt.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            # fig.tight_layout()
            
            if save:
                fig.savefig(f'{folder_path}/{base_file_name}_combined_mask.png')
        
        
        if show:
            plt.show()

    def plot_Vin(self, *, title_addition="", save=False, folder_path=None,
                  base_file_name="", figsize=(3.5,3), ax=None, show=True, color='black',
                  state=[0,0,0,0], Vin=None, partial_N=None, state2=None, color2='orange'):
    
        if ax is None:
            fig, ax = plt.figure(figsize=figsize, dpi=150)
        else:
            fig = ax.figure
        if Vin is None:
            Vin = self.reservoir.sim(obs=state, get_Vin=True)
        
        
        if partial_N is not None:
                Vin_partial = Vin[:partial_N]
        else:
            partial_N = self.reservoir.N
            Vin_partial = Vin
        y_axis = np.repeat(Vin_partial, 1000 // len(Vin_partial))
        x_axis = np.linspace(0, partial_N, len(y_axis))
        ax.plot(x_axis, y_axis, color=color, linewidth=1, linestyle='-')
        ax.text(7, 0.4*360+20, r'$\mathbf{V}^{(0)}_{\text{in}}$', fontsize=8, ha='center', color=color)
        if state2 is not None:
            Vin2 = self.reservoir.sim(obs=state2, get_Vin=True)
            if partial_N is not None:
                    Vin_partial = Vin2[:partial_N]
            else:
                partial_N = self.reservoir.N
                Vin_partial = Vin
            y_axis = np.repeat(Vin_partial, 1000 // len(Vin_partial))
            x_axis = np.linspace(0, partial_N, len(y_axis))
            ax.plot(x_axis, y_axis, color=color2, linewidth=1, linestyle='-', alpha=1)
            ax.text(16, -0.3*360+20, r'$\mathbf{V}^{(1)}_{\text{in}}$', fontsize=8, ha='center', color=color2)
        # ax.set_xlabel('Neuron')
        ax.set_xticks([0, 10, 20])
        ax.set_xticklabels(['0', '10', '20',], fontsize=8)
        # ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, partial_N)
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position('right')
        # ax.set_yticklabels([])

        ax.set_ylabel(r'Voltage $\left[ \text{V} \right]$ ', labelpad=0)
        ax.set_title(r'b) Voltage Input', pad=6, loc='left', fontsize=9) # $\mathbf{V}_{\text{in}}$
        # plt.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        # fig.tight_layout()
        
        if save:
            fig.savefig(f'{folder_path}/{base_file_name}_combined_mask.png')
        if show:
            plt.show()

    def plot_response(self, data, *, points=None, title=None, save=False, 
                      folder_path=None, base_file_name="",
                      overlay_mask=False, Vin=None, show=True, figsize=(10, 5),
                      show_neuron_sat=False, neuron_sat=None, reps, 
                      return_image=False, ax=None, color='royalblue', color2=None,
                      x_on=True, one_y=False):

        if ax is None:
            fig, ax = plt.figure(figsize=figsize, dpi=150)
        else:
            fig = ax.figure

        if folder_path is None and save:
            print('No folder provided, plot will not be saved.')
            save = False
        
        if folder_path is not None:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        x_axis = np.arange(len(data))/self.reservoir.stride
        # x_axis = np.linspace(0, self.reservoir.N, len(data))

        # plt.figure(figsize=figsize)
        if color2 is not None:
            # print(data.shape)
            # print(data[:len(x_axis)//2].shape)
            # print(data[len(x_axis)//2:].shape)
            ax.plot(x_axis[:len(x_axis)//2], data[:len(x_axis)//2], color=color, linewidth=1, zorder=2)
            ax.plot(x_axis[len(x_axis)//2:], data[len(x_axis)//2:], color=color2, linewidth=1, zorder=2)
        else:
            ax.plot(x_axis, data, color=color, label='Response', linewidth=1, zorder=2)

        if points is not None:
            # print(self.reservoir.sd/self.reservoir.stride)
            x_axis = np.arange(len(points))+self.reservoir.sd/self.reservoir.stride
            # print(points)
            # x_axis = np.repeat()
            # x_axis = np.linspace(0+self.reservoir.sd/self.reservoir.stride, self.reservoir.N, len(points))
            ax.scatter(x_axis,points, color='black', marker='o', s=4, zorder=3, label='Neurons')

        # print(len(points))
        if overlay_mask:
            y_axis = np.repeat(self.reservoir.mask.sum(axis=0), len(data) // len(self.reservoir.mask.sum(axis=0)))
            x_axis = np.linspace(0, self.reservoir.N*reps, len(y_axis))
            ax.plot(x_axis, y_axis, label='Combined Mask', color='red', linestyle='--', linewidth=1, zorder=2)
        
        if Vin is not None:
            # print(Vin)
            y_axis = np.repeat(Vin, len(data) // len(Vin))
            x_axis = np.linspace(0, self.reservoir.N*reps, len(y_axis))
            ax.plot(x_axis, y_axis, label='Normed Voltage Input', color='green', linewidth=1, linestyle='--', zorder=0)
        
        if show_neuron_sat:
            ax.text(-1/32*self.reservoir.N, max(data)+.11, f'Neuron Saturation: {neuron_sat:.1f} %', fontsize=10, zorder=4)
        
        ticks = [0, 10, 20, 30, 40]
        tick_labels = [str(int(round(tick*self.reservoir.theta, 0))) for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=8)
        if x_on:
            ax.set_xlabel(r'Non-dimensional Time, $\tilde{t}$', fontsize=8, labelpad=2)
            
            
        # else:
        #     ax.set_xticklabels([], fontsize=8)
        if not one_y:
            ax.set_ylabel(r'MEMS Response, $\tilde{x}$', fontsize=8, labelpad=0)
        ax.set_ylim([-0.5, 1.1])
        ax.set_yticks([-0.5, 0, 0.5, 1])
        ax.set_yticklabels(['-0.5', '0', '0.5', '1'], fontsize=8)
        if title is None:
            ax.set_title('c) Reservoir Response', pad=6, loc='left', fontsize=9)
        else:
            ax.set_title(title, pad=6, loc='left', fontsize=9)
        # plt.legend()
        # plt.tight_layout(pad=.5)
        # ax.axhline(y=0.9, color='red', linestyle='--', zorder=1, linewidth=1)
        ax.axhline(y=0.9, xmin=0, xmax=.05, color='red', linestyle='--', zorder=1, linewidth=1)
        ax.axhline(y=0.9, xmin=.11, xmax=1, color='red', linestyle='--', zorder=1, linewidth=1)
        ax.text(1.35, 0.89, r'$x_s$', fontsize=8, zorder=1, color='red', 
                ha='center', va='center', fontweight='bold')
        vline_color = "#888888"
        x_color = "#444444"
        ax.axvline(x=20, color=vline_color, linestyle='--', zorder=1, linewidth=1)
        ax.text(1.3, -0.4, r'$\mathbf{x}_0$', fontsize=8, zorder=1, color=x_color,ha='center')
        ax.text(21.3, -0.4, r'$\mathbf{x}_1$', fontsize=8, zorder=1, color=x_color,ha='center')
        # ax.annotate('90% Neuron Saturation', xy=(self.reservoir.N/2, 0.9), xytext=(self.reservoir.N/2+2, 0.9), ha='center', va='center', fontsize=7, color='red')
        # plt.text(0, 0.77, r'$x_s$', fontsize=10, zorder=1, color='red')
        ax.grid(True, linestyle='--', alpha=0.5)
        if save:
            plt.savefig(f'{folder_path}/{base_file_name}_sim_response.png')
        if show:
            plt.show()
        if return_image:
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=75)
            buf.seek(0)
            image = Image.open(buf)
            plt.close()
            return image
        else:
            return None

        # plt.figure(figsize=figsize)
        # plt.plot(data)
        # plt.show()
    
    def sim_response(self, *, state=None, seed=None, overlay_mask=False,
                     overlay_Vin=False, overlay_actual_Vin=True,
                     title=None, save=False, folder_path=None, base_file_name="",
                     show=True, figsize=(10, 5), show_neuron_sat=False, reps=1,
                     return_image=False, states=None, ax=None, color='royalblue', color2='royalblue',
                     x_on=True, one_y=False):
        # Vin = np.array([0])
        if state is None:
            rng = np.random.RandomState(seed)
            state = rng.uniform(0, 1, self.reservoir.state_shape)
            # print(f'Random state: {state}')
        
        all_positions = np.array([])
        all_theta_vals = np.array([])
        all_Vin = np.array([])
        for i in range(reps):
            if states is not None:
                state = states[i]
            elif states is None:
                if i>0:
                    state = rng.uniform(0, 1, self.reservoir.state_shape)
            positions, theta_vals, Vin = self.reservoir.sim(obs=state, full_data=True)
            all_positions = np.append(all_positions, positions)
            all_theta_vals = np.append(all_theta_vals, theta_vals)
            all_Vin = np.append(all_Vin, Vin)

        
        # print(all_theta_vals)
        positions = np.reshape(all_positions, (-1, ))
        theta_vals = np.reshape(all_theta_vals, (-1,))
        Vin = np.reshape(all_Vin, (-1,))
        # print(len(theta_vals))

        # positions, theta_vals, Vin = self.reservoir.sim(obs=state, full_data=True)

        neuron_sat = self.reservoir.get_neuron_sat(theta_vals)
        # print(np.mean(positions))
        # print(np.std(positions))

        if not overlay_Vin:
            Vin = None

        if overlay_actual_Vin and overlay_Vin:
            Vin = abs(Vin*self.reservoir.amplification + self.reservoir.VDC)
            Vin = Vin / max(Vin)
            

        image = self.plot_response(positions, points=theta_vals, overlay_mask=overlay_mask,
                           Vin=Vin, title=title, save=save, folder_path=folder_path, 
                           base_file_name=base_file_name, show=show, figsize=figsize,
                           show_neuron_sat=show_neuron_sat, neuron_sat=neuron_sat,
                           reps=reps, return_image=return_image, ax=ax, color=color, color2=color2,
                           x_on=x_on, one_y=one_y)
        return image
    

    def reservoir_subplots(self, *, figsize=(3.5, 2.5), dpi=150,states=None,
                           show=True, save=True, folder_path=None, file_name=None):
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams.update({
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        })
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1,1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0:])

        self.plot_mask(ax=ax1, save=False, show=False, together=False, separate=False, 
                       combined=True, color='black', partial_N=20)
        color='#0072B2'
        color2='#D55E00'
        self.plot_Vin(ax=ax2, save=False, show=False, color=color, color2=color2,state=[0,0,0,0],
                      partial_N=20, state2=[-.1, -0.5, -0.15, -0.5])
        
        self.sim_response(ax=ax3, save=False, show=False, states=[[0,0,0,0], [-0.1, -0.5, -0.15, -0.5]],
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2)


        fig.subplots_adjust(left=0.13,
                            right=0.976,
                            top=0.936,
                            bottom=0.107,
                            wspace=0.438,
                            hspace=0.35,
                            )

        if save:
                fig.savefig(f'{folder_path}/{file_name}.png', dpi=300)
            
        if show:
            plt.show()
        return fig

    def reservoir_response_subplots_3(self, *, figsize=(3.5, 2.5), dpi=150,states=None,
                           show=True, save=True, folder_path=None, file_name=None):
        
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams.update({
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        })
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(3, 1, height_ratios=[1,1,1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        

        
        color='#0072B2'
        color2='#D55E00'

        titles = [r'a) $\theta$ = 0.1', r'b) $\theta$ = 1', r'c) $\theta$ = 2$\pi$']

        self.reservoir = Reservoir(N=20,
                    theta=0.1, #*np.pi,
                    state_shape=4,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [4.8*2,3+4,2*0.418,2+4], #changed
                    norm_offset= [4.8,4,0.5,4],
                    amp=360,
                    VDC=20,
                    tau=0,
                    fb_gain=0,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax1, save=False, show=False, states=[[0,0,0,0], [-0.1, -0.5, -0.15, -0.5]],
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2, x_on=False,
                          title=titles[0], one_y=True)
        self.reservoir = Reservoir(N=20,
                    theta=1, #*np.pi,
                    state_shape=4,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [4.8*2,3+4,2*0.418,2+4], #changed
                    norm_offset= [4.8,4,0.5,4],
                    amp=360,
                    VDC=20,
                    tau=0,
                    fb_gain=0,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax2, save=False, show=False, states=[[0,0,0,0], [-0.1, -0.5, -0.15, -0.5]],
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2, x_on=False,
                          title=titles[1], one_y=True)
        self.reservoir = Reservoir(N=20,
                    theta=2*np.pi,
                    state_shape=4,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [4.8*2,3+4,2*0.418,2+4], #changed
                    norm_offset= [4.8,4,0.5,4],
                    amp=360,
                    VDC=20,
                    tau=0,
                    fb_gain=0,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax3, save=False, show=False, states=[[0,0,0,0], [-0.1, -0.5, -0.15, -0.5]],
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2,
                          title=titles[2], one_y=True)

        fig.subplots_adjust(left=0.13,
                            right=0.976,
                            top=0.936,
                            bottom=0.107,
                            wspace=0.438,
                            hspace=0.552,
                            )
        
        fig.text(0.01, 0.5, r'MEMS Response, $\tilde{x}$',
                 va='center', rotation='vertical')

        if save:
                fig.savefig(f'{folder_path}/{file_name}.png', dpi=300)
            
        if show:
            plt.show()
        return fig

    def reservoir_response_subplots_4(self, *, figsize=(3.5, 2.5), dpi=150,states=None,
                           show=True, save=True, folder_path=None, file_name=None):
        
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams.update({
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        })
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(4, 1, height_ratios=[1,1,1,1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[3, 0])

        
        color='#0072B2'
        color2='#D55E00'

        titles = [r'a) $\theta$ = 0.1', r'b) $\theta$ = 0.5', r'c) $\theta$ = 1', r'd) $\theta$ = 2$\pi$']

        self.reservoir = Reservoir(N=20,
                    theta=0.1, #*np.pi,
                    state_shape=4,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [4.8*2,3+4,2*0.418,2+4], #changed
                    norm_offset= [4.8,4,0.5,4],
                    amp=360,
                    VDC=20,
                    tau=0,
                    fb_gain=0,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax1, save=False, show=False, states=[[0,0,0,0], [-0.1, -0.5, -0.15, -0.5]],
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2, x_on=False,
                          title=titles[0], one_y=True)
        self.reservoir = Reservoir(N=20,
                    theta=0.5, #*np.pi,
                    state_shape=4,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [4.8*2,3+4,2*0.418,2+4], #changed
                    norm_offset= [4.8,4,0.5,4],
                    amp=360,
                    VDC=20,
                    tau=0,
                    fb_gain=0,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax2, save=False, show=False, states=[[0,0,0,0], [-0.1, -0.5, -0.15, -0.5]],
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2, x_on=False,
                          title=titles[1], one_y=True)
        self.reservoir = Reservoir(N=20,
                    theta=1, #*np.pi,
                    state_shape=4,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [4.8*2,3+4,2*0.418,2+4], #changed
                    norm_offset= [4.8,4,0.5,4],
                    amp=360,
                    VDC=20,
                    tau=0,
                    fb_gain=0,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax3, save=False, show=False, states=[[0,0,0,0], [-0.1, -0.5, -0.15, -0.5]],
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2,
                          title=titles[2], x_on=False, one_y=True)
        self.reservoir = Reservoir(N=20,
                    theta=2*np.pi,
                    state_shape=4,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [4.8*2,3+4,2*0.418,2+4], #changed
                    norm_offset= [4.8,4,0.5,4],
                    amp=360,
                    VDC=20,
                    tau=0,
                    fb_gain=0,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax4, save=False, show=False, states=[[0,0,0,0], [-0.1, -0.5, -0.15, -0.5]],
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2,
                          title=titles[3], one_y=True)

        fig.subplots_adjust(left=0.13,
                            right=0.976,
                            top=0.954,
                            bottom=0.087,
                            wspace=0.438,
                            hspace=0.624,
                            )
        fig.text(0.01, 0.5, r'MEMS Response, $\tilde{x}$',
                 va='center', rotation='vertical')
        if save:
                fig.savefig(f'{folder_path}/{file_name}.png', dpi=300)
            
        if show:
            plt.show()
        return fig

    def reservoir_response_subplots_2_mc(self, *, figsize=(3.5, 2.5), dpi=150,states=None,
                           show=True, save=True, folder_path=None, file_name=None):
        
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams.update({
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        })
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(2, 1, height_ratios=[1,1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        
        

        
        color='#0072B2'
        color2='#D55E00'

        titles = [r'a) $\tau$ = 0', r'b) $\tau$ = 20']
        states = [[0,0], [-0.2, -0.05]]

        self.reservoir = Reservoir(N=20,
                    theta=1, #*np.pi,
                    state_shape=2,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [2*1.2, 2*0.07],
                    norm_offset= [1.2, 0.07],
                    amp=360,
                    VDC=20,
                    tau=0,
                    fb_gain=0,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax1, save=False, show=False, states=states,
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2, x_on=False,
                          title=titles[0])
        self.reservoir = Reservoir(N=20,
                    theta=1, #*np.pi,
                    state_shape=2,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [2*1.2, 2*0.07],
                    norm_offset= [1.2, 0.07],
                    amp=360,
                    VDC=20,
                    tau=20,
                    fb_gain=0.5,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax2, save=False, show=False, states=states,
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2,
                          title=titles[1])
        

        fig.subplots_adjust(left=0.13,
                            right=0.976,
                            top=0.933,
                            bottom=0.12,
                            wspace=0.438,
                            hspace=0.3,
                            )

        if save:
                fig.savefig(f'{folder_path}/{file_name}.png', dpi=300)
            
        if show:
            plt.show()
        return fig
    
    def reservoir_response_subplots_3_mc(self, *, figsize=(3.5, 2.5), dpi=150,states=None,
                           show=True, save=True, folder_path=None, file_name=None):
        
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams.update({
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        })
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(3, 1, height_ratios=[1,1,1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        

        
        color='#0072B2'
        color2='#D55E00'

        titles = [r'a) $\theta$ = 0.1', r'b) $\theta$ = 0.5', r'c) $\theta$ = 1']
        states = [[0,0], [-0.2, -0.05]]

        self.reservoir = Reservoir(N=20,
                    theta=0.1, #*np.pi,
                    state_shape=2,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [2*1.2, 2*0.07],
                    norm_offset= [1.2, 0.07],
                    amp=360,
                    VDC=20,
                    tau=0,
                    fb_gain=0,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax1, save=False, show=False, states=states,
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2, x_on=False,
                          title=titles[0],one_y=True)
        self.reservoir = Reservoir(N=20,
                    theta=0.5, #*np.pi,
                    state_shape=2,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [2*1.2, 2*0.07],
                    norm_offset= [1.2, 0.07],
                    amp=360,
                    VDC=20,
                    tau=0,
                    fb_gain=0,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax2, save=False, show=False, states=states,
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2, x_on=False,
                          title=titles[1],one_y=True)
        self.reservoir = Reservoir(N=20,
                    theta=1, #*np.pi,
                    state_shape=2,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [2*1.2, 2*0.07],
                    norm_offset= [1.2, 0.07],
                    amp=360,
                    VDC=20,
                    tau=0,
                    fb_gain=0,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax3, save=False, show=False, states=states,
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2,
                          title=titles[2],one_y=True)

        fig.subplots_adjust(left=0.13,
                            right=0.976,
                            top=0.936,
                            bottom=0.107,
                            wspace=0.438,
                            hspace=0.35,
                            )
        fig.text(0.01, 0.5, r'MEMS Response, $\tilde{x}$',
                 va='center', rotation='vertical')
        
        if save:
                fig.savefig(f'{folder_path}/{file_name}.png', dpi=300)
            
        if show:
            plt.show()
        return fig
    
    def reservoir_response_subplots_4_mc(self, *, figsize=(3.5, 2.5), dpi=150,states=None,
                           show=True, save=True, folder_path=None, file_name=None):
        
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams.update({
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        })
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(4, 1, height_ratios=[1,1,1,1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[3, 0])
        

        
        color='#0072B2'
        color2='#D55E00'

        titles = [r'a) $\theta$ = 0.1', r'b) $\theta$ = 0.5', r'c) $\theta$ = 1',r'd) $\theta$ = 2$\pi$']
        states = [[0,0], [-0.2, -0.05]]

        self.reservoir = Reservoir(N=20,
                    theta=0.1, #*np.pi,
                    state_shape=2,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [2*1.2, 2*0.07],
                    norm_offset= [1.2, 0.07],
                    amp=360,
                    VDC=20,
                    tau=20,
                    fb_gain=0.5,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax1, save=False, show=False, states=states,
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2, x_on=False,
                          title=titles[0],one_y=True)
        self.reservoir = Reservoir(N=20,
                    theta=0.5, #*np.pi,
                    state_shape=2,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [2*1.2, 2*0.07],
                    norm_offset= [1.2, 0.07],
                    amp=360,
                    VDC=20,
                    tau=20,
                    fb_gain=0.5,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax2, save=False, show=False, states=states,
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2, x_on=False,
                          title=titles[1],one_y=True)
        self.reservoir = Reservoir(N=20,
                    theta=1, #*np.pi,
                    state_shape=2,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [2*1.2, 2*0.07],
                    norm_offset= [1.2, 0.07],
                    amp=360,
                    VDC=20,
                    tau=20,
                    fb_gain=0.5,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax3, save=False, show=False, states=states,
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2,
                          title=titles[2], x_on=False,one_y=True)
        
        self.reservoir = Reservoir(N=20,
                    theta=2*np.pi,
                    state_shape=2,
                    mask_seed=2,
                    h=0.02,
                    input_connectivity=0.2,
                    normalize_mask=True,
                    # norm_factor= [4.8*2,3+4,2*0.418,2+4],
                    # norm_offset=[4.8,4,0.5,4],
                    norm_factor= [2*1.2, 2*0.07],
                    norm_offset= [1.2, 0.07],
                    amp=360,
                    VDC=20,
                    tau=20,
                    fb_gain=0.5,
                    sd=0,
                    # load=True,
                    # mask=mask,
                    )
        self.sim_response(ax=ax4, save=False, show=False, states=states,
                          overlay_mask=False, overlay_Vin=False, overlay_actual_Vin=False,
                          show_neuron_sat=False, reps=2,color=color,color2=color2,
                          title=titles[3],one_y=True)

        fig.subplots_adjust(left=0.13,
                            right=0.976,
                            top=0.95,
                            bottom=0.087,
                            wspace=0.438,
                            hspace=0.68,
                            )
        fig.text(0.01, 0.5, r'MEMS Response, $\tilde{x}$',
                 va='center', rotation='vertical')
        if save:
                fig.savefig(f'{folder_path}/{file_name}.png', dpi=300)
            
        if show:
            plt.show()
        return fig