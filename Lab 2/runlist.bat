py .\train.py -r -s sw_ROT_AUG.pth -p aug_plot_ROT.png -e 50 -b 32
py .\train.py -f -s sw_FLIP_AUG.pth -p aug_plot_FLIP.png -e 50 -b 32
py .\train.py -f -r -s sw_BOTH_AUG.pth -p aug_plot_BOTH.png -e 50 -b 32
py .\train.py -s sw_NO_AUG.pth -p aug_plot_NO_AUG.png -e 50 -b 32
