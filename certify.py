import math

import torch

import avocado as avo
import random


def _get_s(rho: avo.core.State, z: str, k: int):
    idx = 0
    x_basis = [avo.core.State(torch.tensor([1 / math.sqrt(2), 1 / math.sqrt(2)])),
               avo.core.State(torch.tensor([1 / math.sqrt(2), -1 / math.sqrt(2)]))]
    y_basis = [avo.core.State(torch.tensor([1 / math.sqrt(2), 1j / math.sqrt(2)])),
               avo.core.State(torch.tensor([1 / math.sqrt(2), -1j / math.sqrt(2)]))]
    z_basis = [avo.core.State(torch.tensor([1, 0])),
               avo.core.State(torch.tensor([0, 1]))]

    pauli_basis = [x_basis, y_basis, z_basis]

    offset = 0
    for i in range(rho.num_qubits):
        idx *= 2
        offset *= 2
        if i < k:
            idx += int(z[i])
        elif i == k:
            offset = 1
        elif i > k:
            idx += int(z[i - 1])
    # offset = 2 ** (rho.num_qubits - k - 1)
    vec = torch.tensor([rho.data[idx], rho.data[idx + offset]])

    # density_mat = torch.conj(vec) @ vec

    a = torch.conj(vec)
    a = a.reshape(2, -1)
    density_mat = torch.matmul(a, vec.reshape(1, 2))
    # random_basis = random.randint(0, 2)

    basis = pauli_basis[random.randint(0, 2)]

    prob = []
    for b in basis:
        a = b.data.reshape(2, -1)
        obv = torch.matmul(a.conj(), b.data.reshape(1, 2))
        p = torch.trace(torch.matmul(obv, density_mat))
        prob.append(abs(p))

    tem_idx = random.choices(range(2), weights=prob)
    return basis[tem_idx[0]]


def _data_acquisition(rho: avo.core.State):
    k = random.randint(0, rho.num_qubits - 1)
    measure_idx = list(range(rho.num_qubits))
    measure_idx.pop(k)
    result = rho.measure(shots=1, qubits_idx=measure_idx)
    # print(result)
    for r in result.keys():
        if result[r] > 0.9:
            return r, k
    assert 0


def _query(psi: avo.core.State, z: str, k: int) -> avo.State:
    idx = 0
    offset = 0
    for i in range(psi.num_qubits):
        idx *= 2
        offset *= 2
        if i < k:
            idx += int(z[i])
        elif i == k:
            offset = 1
            continue
        elif i > k:
            idx += int(z[i - 1])
    # offset = 2 ** (psi.num_qubits - k - 1)
    tem = abs(psi.data[idx]) ** 2 + abs(psi.data[idx + offset]) ** 2
    tem = math.sqrt(tem)

    if tem < 1e-20:
        tem = 1

    data = [psi.data[idx] / tem, psi.data[idx + offset] / tem]
    ret_state = avo.core.State(data=data)
    return ret_state


def certify(rho: avo.core.State, psi: avo.core.State, epsilon: float, T: int, tau: float):
    r""" certify the overlap of these two states >= 1 - epsilon or not,
        Args:
            rho: one of the two states
            psi: one of the two states
            epsilon: error
            T: Samples time
            tau: relaxtion time

        Return:
            if the overlap of rho and psi is lower than 1 - epsilon, return False with high probability
            if the overlap of rho and psi >= 1 - epsilon / (2 * tau), return True with high probability
    """
    # val = psi.bra @ rho.ket @ rho.bra @ psi.ket
    # print(val)
    omegas = []
    if rho.backend != avo.core.Backend.StateVector:
        print('This function only accept state vector currently')
        assert 0
    for _ in range(T):
        sample, k = _data_acquisition(rho)
        psi_kz = _query(psi=psi, k=k, z=sample)

        s = _get_s(rho=rho, k=k, z=sample)
        tem = 3 * s.ket @ s.bra - torch.eye(2)
        omega = psi_kz.bra @ tem @ psi_kz.ket

        omegas.append(omega)

    shadow_overlap = 0
    for o in omegas:
        shadow_overlap += abs(o)
    shadow_overlap /= T

    if abs(shadow_overlap) >= 1 - 0.75 * epsilon / tau:
        return True
    return False


def test():
    epsilon = 0.1
    a = avo.database.state.bell_state(4)
    b = avo.database.state.bell_state(4)
    tem = b.ket @ b.bra
    # print(torch.trace(tem))
    result = a.bra @ tem @ a.ket
    print(result)

    expect = certify(a, b, epsilon=epsilon, T=100, tau=4)
    print(result)

    return


def a_test():
    import math
    x_basis = [avo.core.State(torch.tensor([1 / math.sqrt(2), 1 / math.sqrt(2)])),
               avo.core.State(torch.tensor([1 / math.sqrt(2), -1 / math.sqrt(2)]))]
    y_basis = [avo.core.State(torch.tensor([1 / math.sqrt(2), 1j / math.sqrt(2)])),
               avo.core.State(torch.tensor([1 / math.sqrt(2), -1j / math.sqrt(2)]))]
    z_basis = [avo.core.State(torch.tensor([1, 0])),
               avo.core.State(torch.tensor([0, 1]))]

    xz00 = x_basis[0].kron(z_basis[0])
    xz01 = x_basis[0].kron(z_basis[1])
    xz10 = x_basis[1].kron(z_basis[0])
    xz11 = x_basis[1].kron(z_basis[1])

    measure = avo.loss.Measure(
        measure_basis=[xz00.ket @ xz00.bra, xz01.ket @ xz01.bra, xz10.ket @ xz10.bra, xz11.ket @ xz11.bra])

    idx1 = random.randint(0, 1)
    idx2 = random.randint(0, 1)
    test_state = x_basis[random.randint(0, 1)].kron(y_basis[idx1]).kron(z_basis[idx2])

    results = measure.forward(state=test_state, qubits_idx=[0, 2])

    except_result = torch.tensor([0, 0, 0, 0])
    except_result[idx1 * 2 + idx2] = 1
    err = abs(except_result - results)
    assert torch.max(err) < 1e-7, (
        "The measured results do not match expectations\n"
    )


if __name__ == '__main__':
    a_test()