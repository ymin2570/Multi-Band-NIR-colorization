import torch
import torch.nn as nn
from SENet import SELayer


class OFD(nn.Module):
	"""
	A Comprehensive Overhaul of Feature Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf
	"""
	def __init__(self, in_channels, out_channels):
		super(OFD, self).__init__()
		self.connector = nn.Sequential(*[
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
				nn.InstanceNorm2d(out_channels)
				# nn.BatchNorm2d(out_channels)
			])
		self.meta_net = SELayer(channel=out_channels)

	def forward(self, fm_s, fm_t):
		fm_t = self.connector(fm_t)
		fm_t = self.meta_net(fm_t)
		loss = torch.mean(pow((fm_s - fm_t), 2))  # MSE loss
		return loss


class OFD2(nn.Module):
	"""
	A Comprehensive Overhaul of Feature Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf
	"""
	def __init__(self, in_channels, out_channels):
		super(OFD2, self).__init__()
		self.connector = nn.Sequential(*[
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
				nn.InstanceNorm2d(out_channels)
				# nn.BatchNorm2d(out_channels)
			])
		self.meta_net = SELayer(channel=out_channels)

	def forward(self, fm_s, fm_t):
		fm_t = self.connector(fm_t)
		# fm_t = self.meta_net(fm_t)
		loss = torch.mean(abs(fm_s - fm_t))  # L1 loss
		return loss


