import sys

import matplotlib.pyplot as plt
import numpy as np
import tinyxsf

filename = sys.argv[1]
elo = float(sys.argv[2])
ehi = float(sys.argv[3])

data = tinyxsf.load_pha(filename, elo, ehi)
plt.imshow(data['RMF'], cmap='viridis')
plt.colorbar()
plt.xlabel('Energy')
plt.ylabel('Energy channel')
plt.savefig(sys.argv[1] + '_rmf.pdf', bbox_inches='tight')
plt.close()

plt.title('exposure: %d area: %f' % (data['src_expo'], data['src_expoarea'] / data['src_expo']))
plt.plot(data['ARF'])
plt.ylabel('Sensitive area')
plt.xlabel('Energy channel')
plt.savefig(sys.argv[1] + '_arf.pdf', bbox_inches='tight')
plt.close()
