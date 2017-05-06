require 'nn'

function ActivationModule()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  net:add(nn.Mean(2))
  return net
end
