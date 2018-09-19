"""
Compare Plot
============

_thumb: .5, .5
"""
import arviz as az

az.style.use('arviz-darkgrid')


model_compare = az.compare({
    'Centered eight schools': az.load_arviz_data('centered_eight'),
    'Non-centered eight schools': az.load_arviz_data('non_centered_eight'),
})

az.plot_compare(model_compare, figsize=(12, 4))
