import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plot
from orthogonal_matrices import random_orthogonal

rcParams.update({'figure.autolayout': True})

############ config
num_runs = 1
num_input = 100
num_output = 400 
rank = 2 # other ranks not supported yet 
num_epochs = 10000
esses = [5]#[10, 5, 3]
s_new = 2. 
init_size = 1e-5
epsilon = 0.01
overlap = 0.85
lr = 1e-3
subsample = 50 # how much to subsample empirical timeseries
second_start_time = 0
second_simple_approx = True
#############
tau = 1./lr 

plot.rcParams.update({'font.size': 15})

def _train(sigma_31, sigma_11, W21, W32, num_epochs, track_mode_alignment=False,
           new_input_modes=None, new_output_modes=None,):
    tracks = {
#        "W21": np.zeros([num_epochs, num_input]),
#        "W32": np.zeros([num_epochs, num_output]),
        "loss": np.zeros([num_epochs+1]),
        "real_S0": np.zeros([num_epochs+1]),
        "real_S1": np.zeros([num_epochs+1]),
        "S0": np.zeros([num_epochs+1]),
        "S1": np.zeros([num_epochs+1])
        }
    if track_mode_alignment:
        tracks["alignment"] = np.zeros([num_epochs+1])
        tracks["alignmentinitold"] = np.zeros([num_epochs+1])
        tracks["alignmentinitnew"] = np.zeros([num_epochs+1])
#        tracks["alignmentdeltanew"] = np.ones([num_epochs+1])
        vec0, vec1 = _get_rep_modes(W21, W32, new_input_modes,
                                    new_output_modes, orthogonal=False)
        vec0_init = vec0
        tracks["alignment"][0] = np.dot(vec0, vec1)
        tracks["alignmentinitold"][0] = np.dot(vec0_init, vec0)
        tracks["alignmentinitnew"][0] = np.dot(vec0_init, vec1)

    l = sigma_31 - np.dot(W32, np.dot(W21, sigma_11))
    tracks["loss"][0] = np.sum(np.square(l))
    a00, b00, a01, b01 = _coefficients_from_weights_and_modes(W21, W32, new_input_modes, new_output_modes)
    tracks["S0"][0] = a00 * b00 
    tracks["S1"][0] = a01 * b01
    S21 = np.linalg.svd(W21, compute_uv=False)
    S32 = np.linalg.svd(W32, compute_uv=False)
    tracks["real_S0"][0] = S21[0] * S32[0] 
    tracks["real_S1"][0] = S21[1] * S32[1] 
    old_vecs = []
    for epoch in range(1, num_epochs + 1):
        l = sigma_31 - np.dot(W32, np.dot(W21, sigma_11))
        W21 += lr * np.dot(W32.transpose(), l) 
        W32 += lr * np.dot(l, W21.transpose()) 
#        tracks["W21"][epoch, :] = W21
#        tracks["W32"][epoch, :] = W32
        tracks["loss"][epoch] = np.sum(np.square(l))
        if track_mode_alignment and epoch % subsample == 0:
            vec0, vec1 = _get_rep_modes(W21, W32, new_input_modes,
                                        new_output_modes, orthogonal=False)
            old_vecs.append(vec1)

            tracks["alignment"][epoch] = np.dot(vec0, vec1)
            tracks["alignmentinitold"][epoch] = np.dot(vec0_init, vec0)
            tracks["alignmentinitnew"][epoch] = np.dot(vec0_init, vec1)
            a00, b00, a01, b01 = _coefficients_from_weights_and_modes(W21, W32, new_input_modes, new_output_modes)
            S21 = np.linalg.svd(W21, compute_uv=False)
            S32 = np.linalg.svd(W32, compute_uv=False)
            tracks["real_S0"][epoch] = S21[0] * S32[0] 
            tracks["real_S1"][epoch] = S21[1] * S32[1] 
            tracks["S0"][epoch] = a00*b00
            tracks["S1"][epoch] = a01*b01
#            if len(old_vecs) < 50:
#                old_vec1 = vec1
#            else:
#                old_vec1 = old_vecs.pop(0)
#            tracks["alignmentdeltanew"][epoch] = np.dot(old_vec1, vec1)

    return W21, W32, tracks

def _ugly_function(sc, c0, s, theta):
#    return np.log((sc + c0 + s*np.tanh(theta/2.))/(sc - (c0 + s*np.tanh(theta/2.))))
    return np.arctanh((c0 + s*np.tanh(theta/2.))/sc) # simpler form than the one given in Saxe et al. 

def _estimated_learning_time(a0, b0, s, epsilon, tau):
    if s > 0:
        c0 = 0.5*np.abs(a0**2 - b0**2) 
        theta0 = np.arcsinh(a0*b0/c0)
        thetaf = np.arcsinh((1-epsilon) * s/c0)

        sc = np.sqrt((c0)**2 + (s)**2)

        t = (tau/sc)*(_ugly_function(sc, c0, s, thetaf) - _ugly_function(sc, c0, s, theta0))
    elif s == 0:
        c0 = 0.5*np.abs(a0**2 - b0**2) 
        theta0 = np.arcsinh(a0*b0/c0)
        thetaf = np.arcsinh(epsilon * a0 * b0 /c0)
        t = (tau/(2*c0))*(np.log(np.tanh(theta0/2.)) - np.log(np.tanh(thetaf/2.)))
    else: 
        raise ValueError("Singular values cannot be negative")
    return t

def _estimated_learning_times(a0, b0, s, tau, num_points=5000):
    if s > 0:
        start = a0*b0/s
        end = 1.0
    elif s == 0:
        start = 1.0 
        end = 0.
    else: 
        raise ValueError("Singular values cannot be negative")
    alignments = np.arange(start, end, (1./num_points) *(end-start) ) 
    epsilons = 1.-alignments
    times = np.zeros(num_points)
    for i in range(1, len(alignments)):
        times[i] = _estimated_learning_time(a0, b0, s, epsilons[i], tau)

    if s == 0:
        times = times[1:]
        epsilons = epsilons[1:]
    
    return times, epsilons

def _get_rep_modes(W21, W32, new_input_modes, new_output_modes, orthogonal=True):
    a0s = np.dot(W21, new_input_modes.transpose())
    b0s = np.dot(W32.transpose(), new_output_modes.transpose()) 
    vec0 = np.sqrt(a0s[:, 0] * b0s[:, 0])
    vec0 *= np.sign(a0s[:, 0])
    vec0 /= np.linalg.norm(vec0)
    if orthogonal:
        vec1 = np.copy(vec0)[::-1] 
        vec1[1] *= -1
    else:
        vec1 = np.sqrt(a0s[:, 1] * b0s[:, 1])
        vec1 *= np.sign(a0s[:, 1])
        vec1 /= np.linalg.norm(vec1)
    return vec0, vec1

#def _coefficients_from_weights_and_modes(W21, W32, new_input_modes, new_output_modes):
#    a0s = np.dot(W21, new_input_modes.transpose())
#    b0s = np.dot(W32.transpose(), new_output_modes.transpose()) 
#    vec0, vec1 = _get_rep_modes(W21, W32, new_input_modes, new_output_modes, False)
#    a00 = np.sum((a0s[:, 0]**2))
#    a00 = np.sign(a00) * np.sqrt(np.abs(a00))
#    b00 = np.sum((b0s[:, 0]**2))
#    b00 = np.sign(b00) * np.sqrt(np.abs(b00))
#    index2 = 2 if new_mode == "orthogonal" else 1
#    a01 = np.sum((a0s[:, index2]**2))
#    a01 = np.sign(a01) * np.sqrt(np.abs(a01))
#    b01 = np.sum((b0s[:, index2]**2))
#    b01 = np.sign(b01) * np.sqrt(np.abs(b01))
#    return a00, b00, a01, b01

def _coefficients_from_weights_and_modes(W21, W32, new_input_modes, new_output_modes):
    U21, S21, V21 = np.linalg.svd(W21, full_matrices=False)
    U32, S32, V32 = np.linalg.svd(W32, full_matrices=False)
    a0s = np.dot(V21[:rank, :], new_input_modes.transpose())
    b0s = np.dot(U32[:, :rank].transpose(), new_output_modes.transpose()) 
    index2 = 2 if new_mode == "orthogonal" else 1
    # these abs calls don't generalize to the case that the modes actually start
    # with alignment less than 0, but save me calculating the alignment
    # in the rep space
    a00 = S21[0] * np.abs(a0s[0, 0])
    b00 = S32[0] * np.abs(b0s[0, 0])
    a01 = S21[1] * np.abs(a0s[1, index2]) 
    b01 = S32[1] * np.abs(b0s[1, index2])

    return a00, b00, a01, b01

for run_i in range(num_runs):
    for new_mode in ["orthogonal", "partially_aligned"]:
        for s in esses:
            np.random.seed(run_i) # reproducibility
            new_input_modes = random_orthogonal(num_input)[0:3, :]
            new_output_modes = random_orthogonal(num_output)[0:3, :]
            original_input_mode = overlap * new_input_modes[0:1, :] + np.sqrt(1-overlap**2) *  new_input_modes[1:2, :]
            original_output_mode = overlap * new_output_modes[0:1, :] + np.sqrt(1-overlap**2) *  new_output_modes[1:2, :]
            if new_mode == "orthogonal":
                 S_new =  np.diag([s, 0, s_new])
            else:
                 S_new =  np.diag([s, s_new, 0])

            input_data = np.eye(num_input) # random_orthogonal(num_input)

            sigma_31 = np.dot(original_output_mode.transpose(), s*np.dot(original_input_mode, input_data))
            new_sigma_31 = np.dot(new_output_modes.transpose(), np.dot(S_new, np.dot(new_input_modes, input_data)))
            sigma_11 = np.eye(num_input)
            
            # initial weights
            W21 = np.sqrt(init_size)*random_orthogonal(num_input)[:rank, :]
            W32 = np.sqrt(init_size)*random_orthogonal(num_output)[:, :rank]

            # learning from random init -- theory
            a0s = np.dot(W21, original_input_mode.transpose())
            b0s = np.dot(W32.transpose(), original_output_mode.transpose()) 
            a0 = np.sqrt(np.sum(a0s**2))
            b0 = np.sqrt(np.sum(b0s**2))
            est1_times, est1_epsilons = _estimated_learning_times(a0, b0, s, tau)
            est1_init_loss = s**2 # initial outputs ~= 0 

            # learning from random init -- empirical
            W21, W32, first_tracks = _train(sigma_31, sigma_11, W21, W32, num_epochs,
                                            True,
                                            np.concatenate([original_input_mode, np.zeros_like(original_input_mode), np.zeros_like(original_input_mode)]),
                                            np.concatenate([original_output_mode, np.zeros_like(original_output_mode), np.zeros_like(original_output_mode)]))   
#        print(est1)
#        est1_int = int(est1)
#        print(np.dot(first_tracks["W21"][est1_int, :], original_mode))
#        print(first_tracks["W32"][est1_int, :])
#        print(s * (1-epsilon))
            
        
            # updating to new situation -- theory
            print("debug")
            bbb1 = new_input_modes.copy()
            bbb1[0, :] = original_input_mode
            bbb2 = new_output_modes.copy()
            bbb2[0, :] = original_output_mode
            a0_orig, b0_orig, _, _ = _coefficients_from_weights_and_modes(W21, W32, bbb1, bbb2)
            print(a0_orig, b0_orig)
            a00, b00, a01, b01 = _coefficients_from_weights_and_modes(W21, W32, new_input_modes, new_output_modes)
            est2_0_times, est2_0_epsilons = _estimated_learning_times(a00, b00, s, tau)
            est2_0b_times, est2_0b_epsilons = _estimated_learning_times(np.sqrt(a0_orig**2-a00**2), np.sqrt(a0_orig**2-b00**2), 0, tau)

            est2_1_times, est2_1_epsilons = _estimated_learning_times(a01, b01, s_new, tau)
            
            est2_0_init_loss = s**2 *2 if new_mode == "orthogonal" else 2 *(s**2 - s * s_new) 
            est2_1_init_loss = (s_new)**2


            if second_start_time > 0 and not second_simple_approx:
                print("Staggering theory starts")
                # updating to new situation --empirical  

                W21, W32, second_tracks = _train(new_sigma_31, sigma_11, W21, W32, second_start_time,
                                                 True, new_input_modes, new_output_modes)

                # updating to new situation -- semi-theory starting from after first mode learning is approximately done
                # still isn't perfect because modes aren't truly orthogonal until too late in the learning process
                _, _, a01, b01 = _coefficients_from_weights_and_modes(W21, W32, new_input_modes, new_output_modes)
                est2_1_times, est2_1_epsilons = _estimated_learning_times(a01, b01, s_new, tau)
                est2_1_times += second_start_time # offset
                
                # get rid of gap in plot
                est2_1_times = np.concatenate([[0], est2_1_times])
                est2_1_epsilons = np.concatenate([[1.], est2_1_epsilons])

                # updating to new situation --empirical  
                W21, W32, second_tracks_2 = _train(new_sigma_31, sigma_11, W21, W32, num_epochs-second_start_time,
                                                   True, new_input_modes, new_output_modes)

                for key in second_tracks.keys():
                    second_tracks[key] = np.concatenate([second_tracks[key], second_tracks_2[key][1:]], 0)
    
                staggered_string = "_staggered" # appended to filenames
            else:
                print("Not staggering theory starts")

                if second_simple_approx:
                    # updating to new situation -- second mode theory based on orthogonal condition initial_values 
                    est2_1_times, est2_1_epsilons = _estimated_learning_times(-4.36e-5, -3.51e-5, s_new, tau)

                else:
                    # updating to new situation -- second mode theory 
                    est2_1_times, est2_1_epsilons = _estimated_learning_times(a01, b01, s_new, tau)

                # updating to new situation --empirical  
                W21, W32, second_tracks = _train(new_sigma_31, sigma_11, W21, W32, num_epochs,
                                                 True, new_input_modes, new_output_modes)

                staggered_string = ""
            a00, b00, a01, b01 = _coefficients_from_weights_and_modes(W21, W32, new_input_modes, new_output_modes)


            # plotting
            print(est2_0_epsilons[0])
            print(second_tracks["loss"][0]-4)
            adjusting_loss = est2_0_epsilons*est2_0_init_loss 
            new_loss = est2_1_epsilons**2*est2_1_init_loss
            epochs = range(num_epochs + 1)
            approx_summed_loss = np.zeros_like(epochs, np.float32)  
            adjusting_alignment = np.zeros_like(epochs, np.float32)  
#            blah = np.zeros_like(epochs, np.float32)  
#            blah2 = np.zeros_like(epochs, np.float32)  
#            blah3 = np.zeros_like(epochs, np.float32)  
#            blah4 = np.zeros_like(epochs, np.float32)  
            for i, epoch in enumerate(epochs):
                this_index = np.argmin(np.abs(est2_0_times- epoch)) 
                approx_summed_loss[i] = adjusting_loss[this_index] 
                this_index_2 = np.argmin(np.abs(est2_1_times- epoch)) 
                approx_summed_loss[i] += new_loss[this_index_2] 
#                blah[i] = (1-est2_0_epsilons[this_index])*s 
#                this_index_3 = np.argmin(np.abs(est2_0b_times- epoch)) 
#                blah[i] += est2_0b_epsilons[this_index_3]*s*(1-overlap**2) 
#                q = (1-est2_0_epsilons[this_index])
#                blah[i] = s**2 + second_tracks["real_S0"][i]**2 - 2*s * second_tracks["real_S0"][i] * q
#                blah3[i] = blah[i] - est2_0_epsilons[this_index] * est2_0_init_loss
#                blah4[i] = (2*q - 1) * s 
#                blah2[i] = (1-est2_0_epsilons[this_index])*second_tracks["real_S0"][i]

            # initial learning: loss
            plot.figure()
            plot.plot(epochs[::subsample], first_tracks["loss"][::subsample], ".")
            plot.plot(est1_times, est1_epsilons**2*est1_init_loss)
            plot.xlabel("Epoch")
            plot.ylabel("Loss (initial learning)")
            plot.legend(["Empirical", "Theory"])
            plot.savefig("results/singular_value_%.2f_condition_%s_initial_learning%s.eps" % (s, new_mode, staggered_string))
            plot.figure()

            # initial learning: projections
            plot.figure()
            plot.plot(epochs[::subsample], first_tracks["S0"][::subsample], ".")
            plot.plot(est1_times, (1-est1_epsilons) * s)
            #plot.plot(epochs[::subsample], second_tracks["S0"][::subsample] / second_tracks["real_S0"][::subsample], ".")
            plot.xlabel("Epoch")
            plot.ylabel("Projection strength (initial learning)")
            plot.legend(["Empirical", "Theory"])
            #plot.legend(["Empirical", "Theory", "Alignment"])
            plot.savefig("results/singular_value_%.2f_condition_%s_initial_learning_by_mode%s.eps" % (s, new_mode, staggered_string))
            plot.figure()

            # adjusting : loss
            plot.plot(epochs[::subsample], second_tracks["loss"][::subsample], ".")
            plot.plot(est2_0_times, adjusting_loss)
            plot.plot(est2_1_times, new_loss)
            plot.plot(epochs, approx_summed_loss)
#            plot.plot(epochs[::subsample], blah[::subsample])
#            plot.plot(epochs[::subsample], blah3[::subsample])
            plot.xlabel("Epoch")
            plot.ylabel("Loss (adjusting)")
#            plot.xlim(-100, 500)
            plot.legend(["Empirical", "Theory (adjusted mode)", "Theory (new mode)", "Theory (total)"])
            plot.savefig("results/singular_value_%.2f_condition_%s_adjusting%s.eps" % (s, new_mode, staggered_string))

            # adjusting: projections  
            # note in paper that first mode projection is scaled by s/s_hat, in order to cancel out the change in s_hat
            plot.figure()
            plot.plot(epochs[::subsample], second_tracks["S0"][::subsample], ".")
            plot.plot(epochs[::subsample], second_tracks["S0"][::subsample] * s / second_tracks["real_S0"][::subsample], ".")
            plot.plot(epochs[::subsample], second_tracks["S1"][::subsample], ".")
            plot.plot(est2_0_times, (1-est2_0_epsilons)*s)
#            plot.plot(est2_0b_times, est2_0b_epsilons*s*(1-overlap**2))
#            plot.plot(epochs[::subsample], blah[::subsample])
#            plot.plot(epochs[::subsample], blah2[::subsample])
#            plot.plot(epochs[::subsample], second_tracks["real_S0"][::subsample])
#            lot.plot(epochs[::subsample], blah4[::subsample])
            plot.plot(est2_1_times, (1-est2_1_epsilons) * s_new)
            #plot.plot(epochs[::subsample], second_tracks["S1"][::subsample]/second_tracks["real_S1"][::subsample], ".")
            #plot.xlim(-500, 10000)
#            plot.xlim(-100, 500)
            plot.xlabel("Epoch")
            plot.ylabel("Projection strength (adjusting)")
            plot.legend(["Empirical (1st mode, unscaled)", "Empirical (1st mode, scaled)", "Empirical (2nd mode)", "Theory (1st)", "Theory (2nd)"], loc=(0.22, 0.46))
            #plot.legend(["Empirical (1st mode, unscaled)", "Empirical (1st mode, scaled)", "Empirical (2nd mode)", "Theory (1st)", "Theory (2nd)", "alignment (2nd)"], loc=1)
            plot.savefig("results/singular_value_%.2f_condition_%s_adjusting_by_mode%s.eps" % (s, new_mode, staggered_string))
            plot.figure()

            # discrepancy without scaling
            plot.figure()
            plot.plot(epochs[::subsample], second_tracks["S0"][::subsample], ".")
            plot.plot(epochs[::subsample], second_tracks["S0"][::subsample] * s / second_tracks["real_S0"][::subsample], ".")
            plot.plot(epochs[::subsample], second_tracks["real_S0"][::subsample]/s, ".")
            plot.plot(est2_0_times, (1-est2_0_epsilons)*s)
#            plot.plot(est2_0b_times, est2_0b_epsilons*s*(1-overlap**2))
#            plot.plot(epochs[::subsample], blah[::subsample])
#            plot.plot(epochs[::subsample], blah2[::subsample])
#            plot.plot(epochs[::subsample], second_tracks["real_S0"][::subsample])
#            lot.plot(epochs[::subsample], blah4[::subsample])
#            plot.plot(epochs[::subsample], second_tracks["real_S0"][::subsample])
            #plot.xlim(-500, 10000)
            plot.xlim(-100, 2000)
            plot.ylim(-0.2, 5.2)
            plot.xlabel("Epoch")
            plot.ylabel("Projection strength (adjusting)")
            plot.legend(["Empirical (1st mode, unscaled)", "Empirical (1st mode, scaled)", "s ratio", "Theory (1st)"], loc=5)
            plot.savefig("results/singular_value_%.2f_condition_%s_adjusting_first_mode_discrepancy%s.eps" % (s, new_mode, staggered_string))
            plot.figure()

            plot.figure()
            plot.plot(epochs, second_tracks["alignment"], color='#550055')
            plot.xlabel("Epoch")
            plot.ylabel("Empirical representation alignment")
            plot.savefig("results/singular_value_%.2f_condition_%s_adjusting_rep_alignment%s.eps" % (s, new_mode, staggered_string))
            plot.figure()
            plot.plot(epochs[::subsample], second_tracks["alignmentinitold"][::subsample], '.')
            plot.plot(epochs[::subsample], second_tracks["alignmentinitnew"][::subsample], '.')
            plot.legend(["Adjusted mode", "New mode"])
#            plot.plot(epochs, second_tracks["alignmentdeltanew"])
            plot.xlabel("Epoch")
            plot.ylabel("Empirical representation alignment to initial")
            plot.savefig("results/singular_value_%.2f_condition_%s_adjusting_rep_alignment_2%s.eps" % (s, new_mode, staggered_string))
