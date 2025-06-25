# your_observables.py
import numpy as np

def dphi_jj(leptons, photons, jets, met):
    if len(jets) < 2:
        return 0.0
    phi1, phi2 = jets[0].phi, jets[1].phi
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    if jets[0].eta < jets[1].eta:
        dphi *= -1
    return dphi

def nb_jets(leptons, photons, jets, met):
    return len(jets)

def osdf_veto(leptons, photons, jets, met):
    if len(leptons) < 2:
        return 1
    pdgids = sorted([leptons[0].pdgid, leptons[1].pdgid])
    return 0 if pdgids in [[-11, 13], [-13, 11]] else 1

def define_observables(reader):
    reader.add_observable_from_function('dphi_jj', dphi_jj, required=True)
    reader.add_observable_from_function('n_jets', nb_jets, required=True)
    reader.add_observable_from_function('osdf_veto', osdf_veto, required=True)

    reader.add_cut('dphi_jj > 1.8')
    reader.add_cut('n_jets >= 2')
    reader.add_cut('osdf_veto < 1')
