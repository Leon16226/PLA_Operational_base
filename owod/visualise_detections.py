


def compute_prob(x, distribution):
    eps_radius = 0.5
    num_eval_points = 100
    start_x = x - eps_radius
    end_x = x + eps_radius
    step = (end_x - start_x) / num_eval_points
    dx = torch.linspace(x - eps_radius, x + eps_radius, num_eval_points)
    prob = torch.sum(pdf * step)
    return prob


def update_label_based_on_energy(logits, classes, unk_dist, known_dist):
    unknown_class_index = 80
    cls = classes
    lse = torch.logsumexp(logits[:, :5], dim=1)
    for i, energy in enumerate(lse):
        p_unk = compute_prob(energy, unk_dist)
        p_known = compute_prob(energy, known_dist)


