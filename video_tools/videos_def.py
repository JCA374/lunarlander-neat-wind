"""
Scene list for the explainer. Its narration lives in
video_tools/narration/<slug>.md (one `## <scene>[.n]` section per beat) and is voiced with
the voice-text pipeline into voice/<slug>/.
"""

import scenes as S


def explainer(ctx):
    return [
        S.scene_title(ctx), S.scene_task(ctx), S.scene_reward(ctx), S.scene_network(ctx),
        S.scene_genome(ctx), S.scene_mutation(ctx), S.scene_crossover(ctx), S.scene_speciation(ctx),
        S.scene_loop(ctx), S.scene_fitness(ctx), S.scene_gen_fitness(ctx), S.scene_species(ctx),
        S.scene_gen_ghosts(ctx), S.scene_growth(ctx), S.scene_evolution(ctx), S.scene_read(ctx),
        S.scene_ablation(ctx), S.scene_harder(ctx), S.scene_why(ctx), S.scene_memory(ctx),
        S.scene_compare(ctx), S.scene_outro(ctx),
    ]

