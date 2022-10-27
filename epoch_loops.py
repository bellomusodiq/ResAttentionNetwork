import torch

CUDA = torch.cuda.is_available()


def epoch_train(model, num_epochs, epoch, data_loader, loss_fn, optimizer):
    CUDA = torch.cuda.is_available()
    model.train()
    if epoch % 5 == 0:
        optimizer.param_groups[0]["lr"] *= 0.5
    sum_loss = 0
    sum_accuracy = 0
    iteration = 0
    total_count = 0
    for i, (images, targets) in enumerate(data_loader):
        if CUDA:
            images = images.cuda()
            targets = targets.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        if CUDA:
            sum_accuracy += (targets.cpu() == predicted.cpu()).sum()
        else:
            sum_accuracy += (targets == predicted).sum()
        loss = loss_fn(outputs, targets)
        sum_loss += loss.item() * len(targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration += 1
        total_count += len(images)
        epoch_loss = sum_loss / iteration
        epoch_accuracy = sum_accuracy / total_count

        if i % 10 == 0:
            print(
                "Epoch {}/{}, Iteration: {}/{}, Training loss: {:.3f}, Accuracy: {:.3f}".format(
                    epoch + 1,
                    num_epochs,
                    i,
                    len(data_loader),
                    epoch_loss,
                    epoch_accuracy,
                )
            )
    return epoch_loss, epoch_accuracy


def epoch_test(model, num_epochs, epoch, data_loader, loss_fn):
    CUDA = torch.cuda.is_available()
    model.eval()
    with torch.no_grad():
        sum_loss = 0
        sum_accuracy = 0
        iteration = 0
        total_count = 0
        for i, (images, targets) in enumerate(data_loader):
            if CUDA:
                images = images.cuda()
                targets = targets.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            if CUDA:
                sum_accuracy += (targets.cpu() == predicted.cpu()).sum()
            else:
                sum_accuracy += (targets == predicted).sum()
            loss = loss_fn(outputs, targets)
            sum_loss += loss.item() * len(targets)
            iteration += 1
            total_count += len(images)
            epoch_loss = sum_loss / iteration
            epoch_accuracy = sum_accuracy / total_count

            print(
                "Epoch {}/{}, Iteration: {}/{}, Test loss: {:.3f}, Accuracy: {:.3f}".format(
                    epoch + 1,
                    num_epochs,
                    i,
                    len(data_loader),
                    epoch_loss,
                    epoch_accuracy,
                )
            )
        return epoch_loss, epoch_accuracy
