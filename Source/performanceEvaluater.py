
def load_test(datadir):
        test_transforms = TESTTRANSFORMS
        test_data = torchvision.datasets.ImageFolder(datadir,
								        transform=test_transforms)
        testloader = torch.utils.data.DataLoader(test_data,
								        shuffle=True, batch_size=BATCHSIZE)
        return testloader

def compute_accuracy_and_GAP(net, testloader, return_result):
	net.eval()
	correct = 0
	total = 0
	result = pd.DataFrame(columns = ["pred", "conf", "true"])
	with torch.no_grad():
		for i, (images, labels) in enumerate(testloader):
			images, labels = images.to(device), labels.to(device)
			outputs = net(images)
			confidence, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			temp = pd.DataFrame({"pred":predicted.cpu(), "conf":confidence.cpu(), "true":labels.cpu()})
			result = result.append(temp, ignore_index = True)
	result.sort_values('conf', ascending=False, inplace=True, na_position='last')
	result['correct'] = (result.true == result.pred).astype(int)
	result['prec_k'] = result.correct.cumsum() / (np.arange(len(result)) + 1)
	result['term'] = result.prec_k * result.correct
	gap = result.term.sum() / result.true.count()
	if return_result:
		return correct / total, gap, result
	else:
		return correct / total, gap
