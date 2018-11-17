"""
Styles
======

_thumb: .8, .8
"""
x = np.linspace(0, 1, 100)
dist = stats.beta(2, 5).pdf(x)

style_list = ['default',
              ['default', 'arviz-colors'],
              'arviz-darkgrid',
              'arviz-whitegrid',
              'arviz-white']

for style in style_list:
    with plt.style.context(style):
        plt.figure()
        for i in range(10):
            plt.plot(x, dist - i, f'C{i}', label=f'C{i}')
        plt.title(style)
        plt.xlabel('x')
        plt.ylabel('f(x)', rotation=0, labelpad=15)
        plt.legend(bbox_to_anchor=(1, 1))
