import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from utils.metrics import compute_precision, compute_recall, compute_f1


def train_cumulative(
        args, model, tasks, criterion, optimizer, ewc_regularizer, scheduler, early_stopping,
        fisher_estimate_sample_size=1024,consolidate=True, is_early_stopping=False, is_metric_visualization=True
):
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # criterion = nn.CrossEntropyLoss()
    log_interval = 16

    task_iterations = [
        t.task_epochs * ((len(t.train_dataset) + t.task_batch_size - 1) // t.task_batch_size)
        for t in tasks
    ]

    for task_id, (train_dataset, test_dataset, task_epochs, task_batch_size, input_dim, output_dim) in enumerate(tasks, 1):
        print(f"Training Task {task_id}/{len(tasks)}...")

        model.set_input_output(input_dim, output_dim)
        model = model.to(args.device)

        train_loader = DataLoader(train_dataset, batch_size=task_batch_size, shuffle=True)
        valid_loader = None
        test_loader = DataLoader(test_dataset, batch_size=task_batch_size, shuffle=False)

        iterations_per_epoch = (len(train_dataset) + task_batch_size -1) // task_batch_size

        for epoch in range(1, task_epochs + 1):
            model.train()
            total_train_ce_loss = 0
            total_train_ewc_loss = 0
            total_train_correct = 0
            total_train_samples = 0
            train_progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f'Epoch {epoch}/{task_epochs}', leave=True)
            for batch_idx, (X, y) in train_progress_bar:
                X, y = X.to(args.device), y.to(args.device)

                optimizer.zero_grad()

                logits = model(X)
                train_ce_loss = criterion(logits, y)
                train_ewc_loss = ewc_regularizer(model.named_parameters())
                train_loss = train_ce_loss + train_ewc_loss

                train_loss.backward()
                optimizer.step()

                total_train_ce_loss += (train_ce_loss.item() * y.size(0))
                total_train_ewc_loss += train_ewc_loss.item()
                total_train_correct += (logits.argmax(dim=1) == y).sum().item()
                total_train_samples += y.size(0)

                train_progress_bar.set_postfix({
                    'CE Loss': f'{(total_train_ce_loss / total_train_samples):.4f}',
                    'EWC Loss': f'{(total_train_ewc_loss / batch_idx):.4f}',
                    # 'Train Loss': f'{train_loss.item():.4f}',
                    'Accuracy': f'{(total_train_correct / total_train_samples):.4f}'
                })

                previous_task_iteration = sum(task_iterations[:task_id - 1])
                current_task_iteration = ((epoch - 1) * iterations_per_epoch + batch_idx)
                iteration = previous_task_iteration + current_task_iteration

                if iteration % log_interval == 0:
                    names = [f'task {i + 1}' for i in range(len(tasks))]
                    precisions = [
                        test(
                            args=args,
                            model=model,
                            test_loader=test_loader,
                            criterion=criterion,
                            is_metrics_visualization=False
                        )[1]
                        if i + 1 <= task_id else 0 for i in range(len(tasks))
                    ]
                    title = 'Precision (Consolidated)' if consolidate else 'Precision'
                    # visual.visualize_scalars(vis, precisions, names, title, iteration)

                if iteration % log_interval == 0:
                    title = 'Loss (Consolidated)' if consolidate else 'Loss'
                    # visual.visualize_scalars(vis, [train_loss.item(), train_ce_loss.item(), train_ewc_loss.item()], ['Total', 'CrossEntropy', 'EWC'], title, iteration)

            model.eval()
            total_valid_loss = 0
            total_valid_correct = 0
            total_valid_samples = 0
            valid_progress_bar = tqdm(valid_loader, desc='Evaluating', leave=True)
            with torch.no_grad():
                for X, y in valid_progress_bar:
                    X, y = X.to(args.device), y.to(args.device)

                    logits = model(X)
                    valid_loss = criterion(logits, y)

                    total_valid_loss += (valid_loss.item() * y.size(0))
                    total_valid_correct += (logits.argmax(dim=1) == y).sum().item()
                    total_valid_samples += y.size(0)

                    valid_progress_bar.set_postfix({
                        'Valid Loss': f'{(total_valid_loss / total_valid_samples):.4f}',
                        'Accuracy': f'{(total_valid_correct / total_valid_samples):.4f}'
                    })

            epoch_train_ce_loss = total_train_ce_loss / total_train_samples
            epoch_train_ewc_loss = total_train_ewc_loss / len(train_loader)
            epoch_train_loss = epoch_train_ce_loss + epoch_train_ewc_loss
            epoch_train_accuracy = total_train_correct / total_train_samples
            epoch_valid_loss = total_valid_loss / total_valid_samples
            epoch_valid_accuracy = total_valid_correct / total_valid_samples
            print('Epoch: {} | Train Loss: {:.4f} | Train Acc: {:.4f} | Valid Loss: {:.4f} | Valid Acc: {:.4f}'.format(
                epoch, epoch_train_loss, epoch_train_accuracy, epoch_valid_loss, epoch_valid_accuracy
            ))

            scheduler.step(epoch_valid_loss)

            if is_early_stopping:
                pass

        if consolidate and task_id < len(tasks):
            ewc_regularizer.update_fisher_optimal(
                model,
                current_task_id=task_id,
                data_loader=train_loader,
                sample_size=fisher_estimate_sample_size,
                consolidate=True
            )

        if is_metric_visualization:
            pass


def test(args, model, test_loader, criterion, is_metrics_visualization=True):
    model.eval()
    total_test_loss = 0
    total_test_correct = 0
    total_test_samples = 0
    all_labels = []
    all_predicts = []
    all_predicts_proba = []
    test_progress_bar = tqdm(test_loader, desc='Testing', leave=True)
    with torch.no_grad():
        for X, y in test_progress_bar:
            X, y = X.to(args.device), y.to(args.device)

            logits = model(X)
            test_loss = criterion(logits, y)  # batch average loss

            total_test_loss += (test_loss.item() * y.size(0))
            total_test_correct += (logits.argmax(dim=1) == y).sum().item()
            total_test_samples += y.size(0)

            all_labels.extend(y.cpu().numpy())
            all_predicts.extend(logits.argmax(dim=1).cpu().numpy())
            all_predicts_proba.extend(logits.softmax(dim=1).cpu().numpy())

            test_progress_bar.set_postfix({
                'Test Loss': f'{(total_test_loss / total_test_samples):.4f}',
                'Accuracy': f'{(total_test_correct / total_test_samples):.4f}'
            })

    test_loss = total_test_loss / total_test_samples
    test_accuracy = total_test_correct / total_test_samples
    test_precision = compute_precision(all_labels, all_predicts)
    test_recall = compute_recall(all_labels, all_predicts)
    test_f1 = compute_f1(all_labels, all_predicts)
    print('=======================================================')
    print('Test Loss: {:.4f} | Test Acc: {:.4f}'.format(test_loss, test_accuracy))
    print('Test Precision: {:.4f} | Test Recall (Sensitivity): :{:.4f} | Test F1: {:.4f}'.format(
        test_precision, test_recall, test_f1))
    print('=======================================================')

    if is_metrics_visualization:
        pass

    return test_accuracy, test_precision, test_recall, test_f1