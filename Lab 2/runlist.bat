py .\train.py -f -r -s snoutnet_weights_S.pth -p aug_plotS.png -e 30 -b 128
py .\train.py -f -r -s snoutnet_weights_X.pth -p aug_plotX.png -e 30 -b 64
py .\train.py -n -s snoutnet_weights_T.pth -p aug_plotT.png -e 30 -b 64
py .\train.py -n -s snoutnet_weights_U.pth -p aug_plotU.png -e 30 -b 128
py .\train.py -r -s snoutnet_weights_V.pth -p aug_plotV.png -e 30 -b 64
py .\train.py -r -s snoutnet_weights_W.pth -p aug_plotW.png -e 30 -b 128
