--[[
Implementation of paper: Categorical Reparameterization with Gumbel-Softmax (https://arxiv.org/abs/1611.01144)
--]]

require "nn"
require "nngraph"
do
	
	local GumbelSoftmax = torch.class("GumbelSoftmax")
	local eps = 1e-20

	function GumbelSoftmax:__init() end
	
	-- Gumbel Sample
	function GumbelSoftmax:SampleGumbel(n, m, minVal, maxVal)
		local random_uniform
		if minVal == nil and maxVal == nil then
			random_uniform = torch.rand(n, m)
		else
			random_uniform = minVal + ((maxVal - minVal) * torch.rand(n, m))
		end
		return -torch.log( -torch.log(random_uniform + eps) + eps )
	end
	
	-- Gumbel Softmax Sample
	-- input: logits = logarithm of probability P(X=k), temperature
	-- logits (num_samples, num_categories)
	function GumbelSoftmax:GumbelSoftmaxSample(logits, temperature)
		local G = self:sample_gumbel(logits:size(1), logits:size(2))
		local y = logits + G
		return nn.SoftMax():forward( y/temperature ) 
	end

	-- Gumbel Softmax Sample
	-- param: temperature
	-- input network: logits = logarithm of probability P(X=k), gumbel sample
	-- output: Gumbel softmax sample
	function GumbelSoftmax:GumbelSoftmaxNetwork(temperature)
		local input = -nn.Identity() -- logits
		local noise = -nn.Identity() -- gumbel sample
		local y = {input, noise} - nn.CAddTable() - nn.MulConstant(1/temperature)
		local output = y - nn.SoftMax()
		return nn.gModule({input, noise}, {output})
	end

	-- Gumbel Softmax Sample
	-- input network: logits = logarithm of probability P(X=k), gumbel sample,
	-- 						   temperature(can change in some epochs)
	-- output: Gumbel softmax sample
	function GumbelSoftmax:GumbelSoftmaxNetworkNoParams()
		local input = -nn.Identity() -- logits
		local noise = -nn.Identity() -- gumbel sample
		local temperature = -nn.Identity() -- temperature
		local addition = {input, noise} - nn.CAddTable() 
		local y = {addition, temperature} - nn.CDivTable()
		local output = y - nn.SoftMax()
		return nn.gModule({input, noise}, {output})
	end	

end

