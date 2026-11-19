import unittest
from lab import validate_experiment_config, endpoint_statistics


class ExperimentIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(optimizer_name='adamw', out_dir='runs/adamw-default',
                              learning_rate=.0003, muon_learning_rate=.01,
                              sgdm_learning_rate=.1, beta2=.99, weight_decay=0.,
                              lr_multiplier=1., max_tokens=92160000)

    def test_other_lr_can_change_only_selected_optimizer_rate(self):
        config = {**self.reference, 'optimizer_name':'muon', 'out_dir':'runs/muon-other',
                  'muon_learning_rate':.02}
        validate_experiment_config(config, self.reference, 'muon', 'other')
        config['learning_rate'] = .0006
        with self.assertRaisesRegex(ValueError, 'learning_rate'):
            validate_experiment_config(config, self.reference, 'muon', 'other')

    def test_generic_slide_defaults_do_not_replace_course_beta(self):
        config = {**self.reference, 'beta2':.999}
        with self.assertRaisesRegex(ValueError, 'beta2'):
            validate_experiment_config(config, self.reference, 'adamw', 'default')

    def test_global_lr_multiplier_cannot_change_comparison(self):
        config = {**self.reference, 'lr_multiplier':2.}
        with self.assertRaisesRegex(ValueError, 'lr_multiplier'):
            validate_experiment_config(config, self.reference, 'adamw', 'default')

    def test_other_rate_must_differ_from_default(self):
        config = {**self.reference, 'out_dir':'runs/adamw-other'}
        with self.assertRaisesRegex(ValueError, 'learning rate'):
            validate_experiment_config(config, self.reference, 'adamw', 'other')

    def test_summary_distinguishes_best_point_from_budget_endpoint(self):
        config = dict(batch_size=2, block_size=8, gradient_accumulation_steps=1, max_tokens=32)
        summary = dict(status='completed', tokens=32, step=2, train_seconds=1.,
                       wall_seconds=2., peak_memory_bytes=2*1024**3)
        metrics = [dict(step=1, tokens=16, train_loss=2.1, val_loss=2.2),
                   dict(step=2, tokens=32, train_loss=1.8, val_loss=2.4)]
        stats = endpoint_statistics(config, summary, metrics)
        self.assertEqual(stats['best_recorded_step'], 1)
        self.assertEqual(stats['val_loss'], 2.4)
        self.assertAlmostEqual(stats['endpoint_minus_best'], .2)
        self.assertAlmostEqual(stats['generalization_gap'], .6)
        self.assertEqual(stats['peak_memory_gib'], 2.)
        metrics[-1]['tokens'] = 16
        with self.assertRaisesRegex(ValueError, 'token budget'):
            endpoint_statistics(config, summary, metrics)


if __name__ == '__main__':
    unittest.main()
