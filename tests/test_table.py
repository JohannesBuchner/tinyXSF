import numpy as np
import tinyxsf
from tinyxsf.model import Table, FixedTable
import os
import requests
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

def get_fullfilename(filename, modeldir = os.environ.get('MODELDIR', '.')):
    return os.path.join(modeldir, filename)

def download(url, filename):
    # download file if it does not exist
    fullfilename = get_fullfilename(filename)
    if not os.path.exists(fullfilename):
        print("downloading", url, "-->", fullfilename)
        response = requests.get(url)
        assert response.status_code == 200
        with open(filename, 'wb') as fout:
            fout.write(response.content)

#download('https://zenodo.org/records/1169181/files/uxclumpy-cutoff.fits?download=1', 'uxclumpy-cutoff.fits')
#download('https://zenodo.org/records/2235505/files/wada-cutoff.fits?download=1', 'wada-cutoff.fits')
#download('https://zenodo.org/records/2235457/files/blob_uniform.fits?download=1', 'blob_uniform.fits')
download('https://zenodo.org/records/2224651/files/wedge.fits?download=1', 'wedge.fits')
download('https://zenodo.org/records/2224472/files/diskreflect.fits?download=1', 'diskreflect.fits')


def test_disk_table():
    energies = np.logspace(-0.5, 2, 1000)
    e_lo = energies[:-1]
    e_hi = energies[1:]
    e_mid = (e_lo + e_hi) / 2.0
    deltae = e_hi - e_lo

    tinyxsf.x.abundance("angr")
    tinyxsf.x.cross_section("vern")
    # compare diskreflect to pexmon
    atable = Table(get_fullfilename("diskreflect.fits"))
    Ecut = 400
    Incl = 70
    PhoIndex = 2.0
    ZHe = 1
    ZFe = 1
    for z in 0.0, 1.0, 2.0, 4.0:
        print(f"Case: redshift={z}")
        ftable = FixedTable(get_fullfilename("diskreflect.fits"), energies, redshift=z)
        A = atable(energies, [PhoIndex, Ecut, Incl, z])
        B = tinyxsf.x.pexmon(energies=energies, pars=[PhoIndex, Ecut, -1, z, ZHe, ZFe, Incl]) / (1 + z)**2 / 2
        C = ftable(energies=energies, pars=[PhoIndex, Ecut, Incl])
        l, = plt.plot(e_mid, A / deltae / (1 + z)**2, label="atable")
        plt.plot(e_mid, B / deltae / (1 + z)**2, label="pexmon", ls=':', color=l.get_color())
        plt.xlabel("Energy [keV]")
        plt.ylabel("Spectrum [photons/cm$^2$/s]")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.savefig("pexmon.pdf")
        #plt.close()
        mask = np.logical_and(energies[:-1] > 8 / (1 + z), energies[:-1] < 80 / (1 + z))
        assert_allclose(A[mask], B[mask], rtol=0.1)
        assert_allclose(A, C)

def test_pexpl_table():
    energies = np.logspace(-0.5, 2, 1000)
    e_lo = energies[:-1]
    e_hi = energies[1:]
    e_mid = (e_lo + e_hi) / 2.0
    deltae = e_hi - e_lo
    Ecut = 1000
    Incl = 70
    ZHe = 1
    ZFe = 1
    for PhoIndex in 2.4, 2.0, 1.2:
        for z in 0, 1, 2:
            A = tinyxsf.x.zpowerlw(energies=energies, pars=[PhoIndex, z])
            B = tinyxsf.x.pexmon(energies=energies, pars=[PhoIndex, Ecut, 0, z, ZHe, ZFe, Incl]) / (1 + z)**2
            l, = plt.plot(e_mid * (1 + z), A / deltae, label="atable")
            plt.plot(e_mid * (1 + z), B / deltae / (1 + z)**(PhoIndex - 2), label="pexmon", ls=':', color=l.get_color())
            plt.xlabel("Energy [keV]")
            plt.ylabel("Spectrum [photons/cm$^2$/s]")
            plt.yscale("log")
            plt.xscale("log")
            plt.legend()
            plt.savefig("pexmonpl.pdf")
            assert_allclose(A, B / (1 + z)**(PhoIndex - 2), rtol=0.2, atol=1e-4)

def test_absorber_table():
    tinyxsf.x.abundance("angr")
    tinyxsf.x.cross_section("bcmc")
    # compare uxclumpy to ztbabs * zpowerlw
    atable = Table(get_fullfilename("wedge.fits"))
    PhoIndex = 1.0
    Incl = 80

    for z in 0, 1, 2:
        plt.figure(figsize=(20, 5))
        print("Redshift:", z)
        for elo, NH22 in (0.2, 0.01), (0.3, 0.1), (0.6, 0.4):
        #for elo, NH22 in (0.3, 0.01),:
            energies = np.geomspace(elo / (1 + z), 10, 400)
            e_lo = energies[:-1]
            e_hi = energies[1:]
            e_mid = (e_lo + e_hi) / 2.0
            deltae = e_hi - e_lo

            A = atable(energies, [NH22, PhoIndex, 45.6, Incl, z])
            B = tinyxsf.x.zpowerlw(energies=energies, pars=[PhoIndex, z])
            C = B * tinyxsf.x.zphabs(energies=energies, pars=[NH22, z])
            mask = np.logical_and(energies[:-1] > elo / (1 + z), energies[:-1] / (1 + z) < 80)
            mask[np.abs(energies[:-1] - 6.4 / (1 + z)) < 0.1] = False
            plt.plot(e_mid, A / deltae, label="atable", ls='--', color='k', lw=0.5)
            A[~mask] = np.nan
            B[~mask] = np.nan
            C[~mask] = np.nan
            plt.plot(e_mid, A / deltae, label="atable", color='k')
            plt.plot(e_mid, B / deltae, label="pl", ls="--", color='orange', lw=0.5)
            plt.plot(e_mid, C / deltae, label="pl*tbabs", color='r', lw=1)
            plt.xlabel("Energy [keV]")
            plt.ylabel("Spectrum [photons/cm$^2$/s/keV]")
            plt.ylim(0.04, 3)
            plt.yscale("log")
            plt.xscale("log")
            plt.legend()
            plt.savefig("abspl_z%d.pdf" % z)
            print(energies[np.argmax(np.abs(np.log10(A[mask] / C[mask])))])
            assert_allclose(A[mask], C[mask], rtol=0.2)
        plt.close()
