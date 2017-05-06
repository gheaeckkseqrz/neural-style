require 'activation_module'

-- Define an nn Module to compute style loss in-place
local StyleActivationsLoss, parent = torch.class('nn.StyleActivationsLoss', 'nn.Module')

function StyleActivationsLoss:__init(strength, normalize, iterations)
   parent.__init(self)
   self.normalize = normalize or false
   self.strength = strength
   self.target = nil
   self.loss = 0
   self.mode = 'none'

   self.activation_module = ActivationModule():cuda()
   self.crit = nn.MSECriterion():cuda()
end

function StyleActivationsLoss:updateOutput(input)
   if self.mode == 'capture' then
      self.target = self.activation_module:forward(input):clone()
   elseif self.mode == 'loss' then
      self.G = self.activation_module:forward(input)
      self.loss = self.crit:forward(self.G, self.target)
   end
   self.output = input
   return self.output
end

function StyleActivationsLoss:updateGradInput(input, gradOutput)
   if self.mode == 'loss' then
      local dG = self.crit:backward(self.G, self.target)
      self.gradInput = self.activation_module:backward(input, dG)
      self.gradInput:mul(self.strength)
      self.localGradInput = self.gradInput:clone()

      if self.normalize and torch.norm(self.gradInput, 2) > 1 then
	 self.gradInput:div(torch.norm(self.gradInput, 2))
      end

      self.gradInput:add(gradOutput)
   else
      self.gradInput = gradOutput
   end
   return self.gradInput
end
