import torch
torch.set_default_dtype(torch.float64)

def test_HO2D_slaterdet():
    """ IMPORTANT TEST
        Test the single-particle orbitals constructed in HO2D and the corresponding 
    many-body Slater determinants are indeed eigenfunctions of the 2D harmonic
    oscillator Hamiltonian.
        This function can largely indicate correctness of the primitive
    LogAbsSlaterDet implemented in slater.py, and the class method
    FreeFermion.log_prob implemented in base_dist.py.
    """
    from orbitals import HO2D
    from utils import y_grad_laplacian

    ho2d = HO2D()
    batch = 20

    """ Test the single-particle orbitals. """
    for orbital, E in zip(ho2d.orbitals, ho2d.Es):
        x = torch.randn(batch, 2, requires_grad=True)
        log, grad_log, laplacian_log = y_grad_laplacian(
            lambda x: orbital(x).log(), x)
        kinetic = - 1/2 * laplacian_log - 1/2 * (grad_log**2).sum(dim=-1)
        potential = 0.5 * (x**2).sum(dim=-1)
        Eloc = kinetic + potential
        assert torch.allclose(Eloc, E * torch.ones(batch))

    """ Test a single Slater determinant. """
    from slater import LogAbsSlaterDet
    log_abs_slaterdet = LogAbsSlaterDet.apply

    n = 5
    orbitals, Es = ho2d.fermion_states_random(n)
    x = torch.randn(batch, n, 2, requires_grad=True)
    log, grad_log, laplacian_log = y_grad_laplacian(
        lambda x: log_abs_slaterdet(orbitals, x), x)
    kinetic = - 1/2 * laplacian_log - 1/2 * (grad_log**2).sum(dim=(-2, -1))
    potential = 0.5 * (x**2).sum(dim=(-2, -1))
    Eloc = kinetic + potential
    #print("Eloc:", Eloc, "\nEs:", Es)
    assert torch.allclose(Eloc, sum(Es) * torch.ones(batch))

    """ Test a complete Fermion wavefunction composed of two Slater determinants. """
    from base_dist import FreeFermion
    nup, ndown = 3, 6
    orbitals_up, Es_up = ho2d.fermion_states_random(nup)
    orbitals_down, Es_down = ho2d.fermion_states_random(ndown)

    freefermion = FreeFermion()
    x = torch.randn(batch, nup + ndown, 2, requires_grad=True)
    logp, grad_logp, laplacian_logp = y_grad_laplacian(
        lambda x: freefermion.log_prob(orbitals_up, orbitals_down, x), x)
    kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))
    potential = 0.5 * (x**2).sum(dim=(-2, -1))
    Eloc = kinetic + potential
    #print("Eloc:", Eloc, "\nEs_up:", Es_up, "Es_down:", Es_down)

    assert torch.allclose(Eloc, sum(Es_up + Es_down) * torch.ones(batch))

def test_HO2D_slaterdet_multstates():
    """ IMPORTANT TEST
        Test multiple many-body Slater determinants are indeed eigenfunctions of the
    2D harmonic oscillator Hamiltonian with correct energies.
        This function can largely indicate correctness of the primitive
    LogAbsSlaterDetMultStates implemented in slater.py, and the class method
    FreeFermion.log_prob_multstates implemented in base_dist.py.
    """
    from orbitals import HO2D
    from utils import y_grad_laplacian
    import random

    ho2d = HO2D()
    Nstates = 10
    state_indices_collection = dict( zip(range(Nstates), 
                                         random.choices(range(1, 10), k=Nstates)) )
    print("\nstate_indices_collection:", state_indices_collection)
    batch = sum(times for times in state_indices_collection.values())
    print("batch:", batch)

    """ Test a single Slater determinant."""
    from slater import LogAbsSlaterDetMultStates
    log_abs_slaterdet_multstates = LogAbsSlaterDetMultStates.apply
    n = 5
    states = []
    Es = []
    for _ in range(Nstates):
        orbitals, orbitalEs = ho2d.fermion_states_random(n)
        states.append(orbitals)
        Es.append(sum(orbitalEs))
    print("StateEs:", Es)
    Es = torch.cat(tuple( Es[idx] * torch.ones(times) 
            for idx, times in state_indices_collection.items() ))

    x = torch.randn(batch, n, 2, requires_grad=True)
    log, grad_log, laplacian_log = y_grad_laplacian(
        lambda x: log_abs_slaterdet_multstates(states, state_indices_collection, x), x)
    kinetic = - 1/2 * laplacian_log - 1/2 * (grad_log**2).sum(dim=(-2, -1))
    potential = 0.5 * (x**2).sum(dim=(-2, -1))
    Eloc = kinetic + potential
    print("Eloc:", Eloc, "\nEs:", Es)
    assert torch.allclose(Eloc, Es)
    
    """ Test a complete Fermion wavefunction composed of two Slater determinants. """
    from base_dist import FreeFermion
    nup, ndown = 3, 6
    states = []
    Es = []
    for _ in range(Nstates):
        orbitals_up, orbitalEs_up = ho2d.fermion_states_random(nup)
        orbitals_down, orbitalEs_down = ho2d.fermion_states_random(ndown)
        states.append((orbitals_up, orbitals_down))
        Es.append(sum(orbitalEs_up + orbitalEs_down))
    print("StateEs:", Es)
    Es = torch.cat(tuple( Es[idx] * torch.ones(times) 
            for idx, times in state_indices_collection.items() ))

    freefermion = FreeFermion()
    x = torch.randn(batch, nup + ndown, 2, requires_grad=True)
    logp, grad_logp, laplacian_logp = y_grad_laplacian(
        lambda x: freefermion.log_prob_multstates(states, state_indices_collection, x), x)
    kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))
    potential = 0.5 * (x**2).sum(dim=(-2, -1))
    Eloc = kinetic + potential
    print("Eloc:", Eloc, "\nEs:", Es)
    assert torch.allclose(Eloc, Es)

def test_HO2D_sample():
    from orbitals import HO2D
    from base_dist import FreeFermion
    ho2d = HO2D()

    nup, ndown = 3, 6
    orbitals_up, _ = ho2d.fermion_states_random(nup)
    orbitals_down, _ = ho2d.fermion_states_random(ndown)

    freefermion = FreeFermion()
    batch, n = (20, 30), nup + ndown
    x = torch.randn(*batch, n, 2)

    logp = freefermion.log_prob(orbitals_up, orbitals_down, x)
    assert logp.shape == batch

    samples = freefermion.sample(orbitals_up, orbitals_down, batch)
    assert samples.shape == (*batch, n, 2)

def test_HO2D_sample_multstates():
    """ Test the speed of MC sampling in the case of multiple states."""
    from orbitals import HO2D
    from base_dist import FreeFermion
    from torch.distributions.categorical import Categorical
    from collections import Counter
    import time

    ho2d = HO2D()

    nup, ndown = 6, 0
    deltaE = 3
    batch, n = 8000, nup + ndown

    states = ho2d.fermion_states(nup, ndown, deltaE)
    print("\ndeltaE = %.1f, total number of states = %d" % (deltaE, len(states)))
    state_dist = Categorical(logits=torch.randn(len(states)))
    state_indices = state_dist.sample((batch,))
    state_indices_collection = Counter(sorted(state_indices.tolist()))

    device = torch.device("cuda:1")
    freefermion = FreeFermion(device=device)

    x = torch.randn(batch, n, 2, device=device)
    logp1 = freefermion.log_prob_multstates(states, state_indices_collection, x, method=1)
    assert logp1.shape == (batch,)
    logp2 = freefermion.log_prob_multstates(states, state_indices_collection, x, method=2)
    assert torch.allclose(logp1, logp2)

    for cpu in (True, False):
        for method in (1, 2):
            start = time.time()
            x = freefermion.sample_multstates(states, state_indices_collection, (batch,), 
                                                cpu=cpu, method=method)
            print(("CPU" if cpu else "GPU") + 
                  ", method %d. Time to take (hours per 100 iters):" % method, 
                  (time.time() - start) * 100 / 3600)
            assert x.shape == (batch, n, 2)
