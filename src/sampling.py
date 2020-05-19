import tensorflow as tf
import tensorflow_probability as tfp
import arviz as az

tfd = tfp.distributions
tfb = tfp.bijectors

USE_XLA = False  # @param
NUMBER_OF_CHAINS = 2  # @param
NUMBER_OF_BURNIN = 500  # @param
NUMBER_OF_SAMPLES = 500  # @param
NUMBER_OF_LEAPFROG_STEPS = 4  # @param


def _trace_to_arviz(
    trace=None,
    sample_stats=None,
    observed_data=None,
    prior_predictive=None,
    posterior_predictive=None,
    inplace=True,
):

    if trace is not None and isinstance(trace, dict):
        trace = {k: v.numpy() for k, v in trace.items()}
    if sample_stats is not None and isinstance(sample_stats, dict):
        sample_stats = {k: v.numpy().T for k, v in sample_stats.items()}
    if prior_predictive is not None and isinstance(prior_predictive, dict):
        prior_predictive = {k: v[np.newaxis] for k, v in prior_predictive.items()}
    if posterior_predictive is not None and isinstance(posterior_predictive, dict):
        if isinstance(trace, az.InferenceData) and inplace == True:
            return trace + az.from_dict(posterior_predictive=posterior_predictive)
        else:
            trace = None

    return az.from_dict(
        posterior=trace,
        sample_stats=sample_stats,
        prior_predictive=prior_predictive,
        posterior_predictive=posterior_predictive,
        observed_data=observed_data,
    )


@tf.function(autograph=False, experimental_compile=USE_XLA)
def run_nuts_chain(
    init_state,
    bijectors,
    step_size,
    target_log_prob_fn,
    num_samples=NUMBER_OF_SAMPLES,
    burnin=NUMBER_OF_BURNIN,
):
    def trace_fn(_, pkr):
        return (
            pkr.inner_results.inner_results.target_log_prob,
            pkr.inner_results.inner_results.leapfrogs_taken,
            pkr.inner_results.inner_results.has_divergence,
            pkr.inner_results.inner_results.energy,
            pkr.inner_results.inner_results.log_accept_ratio,
        )

    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size=step_size),
        bijector=bijectors,
    )

    hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=burnin,
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            inner_results=pkr.inner_results._replace(step_size=new_step_size)
        ),
        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    )

    # Sampling from the chain.
    chain_state, sampler_stat = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=burnin,
        current_state=init_state,
        kernel=hmc,
        trace_fn=trace_fn,
    )

    sampler_stat = sampler_stat[
        0
    ]  # for some reason this is different than the HMC sampler
    sampler_stat = sampler_stat[
        0
    ]  # for some reason this is different than the HMC sampler
    return chain_state, sampler_stat


@tf.function(autograph=False, experimental_compile=USE_XLA)
def run_hmc_chain(
    init_state,
    bijectors,
    step_size,
    target_log_prob_fn,
    num_leapfrog_steps=NUMBER_OF_LEAPFROG_STEPS,
    num_samples=NUMBER_OF_SAMPLES,
    burnin=NUMBER_OF_BURNIN,
):
    def _trace_fn_transitioned(_, pkr):
        return pkr.inner_results.inner_results.log_accept_ratio

    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=tf.convert_to_tensor(step_size, dtype=tf.float32),
    )

    inner_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=hmc_kernel, bijector=bijectors
    )

    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel,
        target_accept_prob=0.8,
        num_adaptation_steps=int(0.8 * burnin),
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    )

    results, sampler_stat = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=burnin,
        current_state=init_state,
        kernel=kernel,
        trace_fn=_trace_fn_transitioned,
    )
    return results, sampler_stat


def sample_posterior(
    jdc,
    observed_data,
    params,
    init_state=None,
    bijectors=None,
    step_size=0.1,
    method="hmc",
    num_chains=NUMBER_OF_CHAINS,
    num_samples=NUMBER_OF_SAMPLES,
    burnin=NUMBER_OF_BURNIN,
):

    if init_state is None:
        init_state = list(jdc.sample(num_chains)[:-1])

    if bijectors is None:
        bijectors = [tfb.Identity() for i in init_state]

    target_log_prob_fn = lambda *x: jdc.log_prob(x + observed_data)

    if method == "hmc":
        results, sample_stats = run_hmc_chain(
            init_state,
            bijectors,
            step_size=step_size,
            target_log_prob_fn=target_log_prob_fn,
            num_samples=num_samples,
            burnin=burnin,
        )
    elif method == "nuts":
        results, sample_stats = run_nuts_chain(
            init_state,
            bijectors,
            step_size=step_size,
            target_log_prob_fn=target_log_prob_fn,
            num_samples=num_samples,
            burnin=burnin,
        )
    else:
        raise ValueError("invalid sampling method specified")

    stat_names = ["mean_tree_accept"]
    sampler_stats = dict(zip(stat_names, [sample_stats]))

    transposed_results = []

    for r in results:
        if len(r.shape) == 2:
            transposed_shape = [1, 0]
        elif len(r.shape) == 3:
            transposed_shape = [1, 0, 2]
        else:
            transposed_shape = [1, 0, 2, 3]

        transposed_results.append(tf.transpose(r, transposed_shape))

    posterior = dict(zip(params, transposed_results))

    az_trace = _trace_to_arviz(trace=posterior, sample_stats=sampler_stats)

    return posterior, az_trace
