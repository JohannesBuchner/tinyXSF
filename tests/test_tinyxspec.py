"""Tests for the tinyxspec command line helpers."""
import numpy as np
import pytest

from tinyxsf.tinyxspec import (default_prior, evaluate_model, get_parameter_keys,
                               parse_model_expression, parse_prior, resolve_components,
                               resolve_model_tree, sample_parameters, sample_prior,
                               update_parameter_file)


def test_parse_model_expression():
    assert parse_model_expression('ztbabs*(zpowerlw + pexmon)') == (
        ('*', [('comp', 'ztbabs'), ('+', [('comp', 'zpowerlw'), ('comp', 'pexmon')])]))
    assert parse_model_expression('a-b') == ('-', [('comp', 'a'), ('comp', 'b')])
    assert parse_model_expression(' a * ( b ) ') == ('*', [('comp', 'a'), ('comp', 'b')])
    # chains of the same operator become one n-ary node
    assert parse_model_expression('a*b*c') == (
        ('*', [('comp', 'a'), ('comp', 'b'), ('comp', 'c')]))
    assert parse_model_expression('gsmooth{powerlw}') == (
        ('conv', 'gsmooth', ('comp', 'powerlw')))
    assert parse_model_expression('gsmooth{(a+b)*c}*d') == (
        ('*', [('conv', 'gsmooth', ('*', [
            ('+', [('comp', 'a'), ('comp', 'b')]), ('comp', 'c')])), ('comp', 'd')]))


def test_parse_model_expression_invalid():
    for expression in ['a)b', 'a*(b', 'a$b', '', '2', 'a**b', 'a/b', 'gsmooth{}', 'a b']:
        with pytest.raises(ValueError):
            parse_model_expression(expression)


def test_resolve_components():
    assert resolve_components(['ztbabs', 'ZPOWERLW']) == ['zTBabs', 'zpowerlw']
    # unique prefixes are accepted as abbreviations, like in XSPEC
    assert resolve_components(['ztb', 'pexmon'])[0] == 'zTBabs'


def test_resolve_model_tree_numbering():
    # repeated components are numbered breadth-first in XSPEC style
    tree = parse_model_expression('zpowerlw + zpowerlw*zpowerlw')
    occurrences, node_keys, npar = resolve_model_tree(tree)
    assert [occ['compkey'] for occ in occurrences] == ['zpowerlw', 'zpowerlw_2', 'zpowerlw_3']
    assert npar == 3 * 3
    tree = parse_model_expression('ztbabs*(zpowerlw + pexmon)')
    occurrences, node_keys, npar = resolve_model_tree(tree)
    assert [occ['compkey'] for occ in occurrences] == ['zTBabs', 'zpowerlw', 'pexmon']
    # the normalisation is an extra parameter of non-convolution models
    assert occurrences[0]['has_norm'] and occurrences[0]['nparams'] == 3


def test_parse_prior():
    assert parse_prior('0,10,uniform') == dict(kind='uniform', min=0.0, max=10.0)
    assert parse_prior('1e-10,10,loguniform') == dict(kind='loguniform', min=1e-10, max=10.0)
    assert parse_prior('0,10,normal,1.9,0.15') == dict(
        kind='normal', mean=1.9, sigma=0.15, min=0.0, max=10.0)
    assert parse_prior('0,10,lognormal,-1,0.5') == dict(
        kind='lognormal', mean=-1.0, sigma=0.5, min=0.0, max=10.0)
    assert parse_prior('0,10,fixed,1000') == dict(kind='fixed', value=1000.0, min=0.0, max=10.0)
    assert parse_prior('0,10,linked,ztbabs:nH') == dict(
        kind='linked', target='ztbabs:nH', min=0.0, max=10.0)


def test_parse_prior_invalid():
    for spec in ['10,0,uniform', '-1,10,loguniform', '0,10,normal,1.9', '0,10,weibull,1,2',
                 '0,10', 'uniform', 'a,b,uniform', '0,10,fixed', '0,10,linked', 'nonsense']:
        with pytest.raises(ValueError):
            parse_prior(spec)


def test_sample_prior():
    rng = np.random.default_rng(42)
    values = sample_prior(parse_prior('2,3,uniform'), rng, 20)
    assert values.shape == (20,)
    assert np.all(values >= 2) and np.all(values <= 3)
    rng2 = np.random.default_rng(42)
    np.testing.assert_allclose(values, sample_prior(parse_prior('2,3,uniform'), rng2, 20))
    np.testing.assert_allclose(sample_prior(parse_prior('0,0,fixed,3'), rng, 5), np.full(5, 3.0))
    # log-uniform samples are log-symmetric around sqrt(min*max)
    values = sample_prior(parse_prior('1e-4,1,loguniform'), np.random.default_rng(1), 1000)
    assert values.min() >= 1e-4 and values.max() <= 1


def test_sample_parameters_linked():
    keys = ['zTBabs:nH', 'zpowerlw:PhoIndex', 'zpowerlw:norm']
    priors = {'zTBabs:nH': '0,10,uniform', 'zpowerlw:PhoIndex': '0,0,linked,zTBabs:nH',
              'zpowerlw:norm': '1,2,uniform'}
    samples = sample_parameters(keys, priors, np.random.default_rng(3), 10)
    assert samples.shape == (10, 3)
    np.testing.assert_allclose(samples[:, 0], samples[:, 1])
    # chains of links are followed
    priors = {'zTBabs:nH': '0,0,linked,zpowerlw:norm',
              'zpowerlw:PhoIndex': '0,0,linked,zTBabs:nH', 'zpowerlw:norm': '1,2,uniform'}
    samples = sample_parameters(keys, priors, np.random.default_rng(3), 10)
    np.testing.assert_allclose(samples[:, 2], samples[:, 0])
    np.testing.assert_allclose(samples[:, 2], samples[:, 1])
    # circular links raise
    priors = {'zTBabs:nH': '0,0,linked,zpowerlw:PhoIndex',
              'zpowerlw:PhoIndex': '0,0,linked,zTBabs:nH', 'zpowerlw:norm': '1,2,uniform'}
    with pytest.raises(ValueError):
        sample_parameters(keys, priors, np.random.default_rng(3), 10)
    # links to unknown parameters raise
    priors = {'zTBabs:nH': '0,0,linked,nosuch:param', 'zpowerlw:PhoIndex': '1,2,uniform',
              'zpowerlw:norm': '1,2,uniform'}
    with pytest.raises(ValueError):
        sample_parameters(keys, priors, np.random.default_rng(3), 10)


def test_default_prior():
    import xspec_models_cxc as x
    mul = x.ModelType.Mul
    add = x.ModelType.Add
    assert default_prior(mul, ('nH', 1.0, 0.0, 100000.0, False)) == '0.0,100000.0,uniform'
    assert default_prior(mul, ('Redshift', 0.0, -0.999, 10.0, True)) == '-0.999,10.0,fixed,0.0'
    assert default_prior(mul, ('norm', 1.0, 0.0, 10.0, False)) == '0.5,2,uniform'
    assert default_prior(add, ('norm', 1.0, 0.0, 10.0, False)) == '1e-10,10,loguniform'


def test_update_parameter_file(tmp_path):
    def components_keys(model):
        tree = parse_model_expression(model)
        occurrences, node_keys, npar = resolve_model_tree(tree)
        return get_parameter_keys(occurrences)

    filename = str(tmp_path / 'parameters.txt')
    model = 'ztbabs*(zpowerlw + pexmon)'
    keys, default_priors = components_keys(model)

    priors, stats = update_parameter_file(filename, model, keys, default_priors)
    assert stats['created']
    assert keys[0] == 'zTBabs:nH' and keys[-1] == 'pexmon:norm'
    assert len(keys) == 14
    lines = open(filename).read().splitlines()
    assert lines[0] == model
    assert len(lines) == 1 + len(keys)

    # user edits a prior; rerunning with the same model reuses it
    lines[1] = 'zTBabs:nH 0,5,uniform'
    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    priors, stats = update_parameter_file(filename, model, keys, default_priors)
    assert not stats['created'] and not stats['commented_out'] and not stats['added']
    assert priors['zTBabs:nH'] == '0,5,uniform'

    # switching to a model without pexmon comments out its parameters,
    # keeps the other priors and records the previous model
    model2 = 'ztbabs*zpowerlw'
    keys2, default_priors2 = components_keys(model2)
    priors, stats = update_parameter_file(filename, model2, keys2, default_priors2)
    assert sorted(stats['commented_out']) == sorted(
        ['pexmon:PhoIndex', 'pexmon:foldE', 'pexmon:rel_refl', 'pexmon:redshift',
         'pexmon:abund', 'pexmon:Fe_abund', 'pexmon:Incl', 'pexmon:norm'])
    assert priors['zTBabs:nH'] == '0,5,uniform'
    lines = open(filename).read().splitlines()
    assert lines[0] == model2
    assert lines[1] == '# %s  # previous model' % model
    assert any(line.startswith('# pexmon:norm ') for line in lines)

    # switching back reactivates the commented-out entries with their priors
    priors, stats = update_parameter_file(filename, model, keys, default_priors)
    assert sorted(stats['reactivated']) == sorted(
        ['pexmon:PhoIndex', 'pexmon:foldE', 'pexmon:rel_refl', 'pexmon:redshift',
         'pexmon:abund', 'pexmon:Fe_abund', 'pexmon:Incl', 'pexmon:norm'])
    assert not stats['added']
    lines = open(filename).read().splitlines()
    assert lines[0] == model
    assert any(line == 'pexmon:PhoIndex 1.1,2.5,uniform' for line in lines)


def test_evaluate_model():
    tree = parse_model_expression('ztbabs*zpowerlw')
    occurrences, node_keys, npar = resolve_model_tree(tree)
    keys, default_priors = get_parameter_keys(occurrences)
    energies = np.geomspace(0.1, 50, 100)
    priors = {key: default_priors[key] for key in keys}
    priors['zTBabs:nH'] = '0,1,uniform'
    priors['zTBabs:norm'] = '0.5,2,uniform'
    priors['zpowerlw:norm'] = '1,10,loguniform'
    samples = sample_parameters(keys, priors, np.random.default_rng(0), 4)
    assert samples.shape == (4, npar)
    spectra = evaluate_model(tree, occurrences, node_keys, samples, energies)
    assert spectra.shape == (4, len(energies) - 1)
    assert np.all(np.isfinite(spectra))
    assert np.all(spectra >= 0)


def test_evaluate_model_convolution():
    tree = parse_model_expression('gsmooth{zpowerlw}')
    occurrences, node_keys, npar = resolve_model_tree(tree)
    keys, default_priors = get_parameter_keys(occurrences)
    energies = np.geomspace(0.1, 50, 100)
    samples = sample_parameters(
        keys, {key: default_priors[key] for key in keys}, np.random.default_rng(0), 4)
    spectra = evaluate_model(tree, occurrences, node_keys, samples, energies)
    assert spectra.shape == (4, len(energies) - 1)
    assert np.all(np.isfinite(spectra))
    # gsmooth edge undershoot is at floating point level
    assert spectra.min() > -1e-10 * spectra.max()
