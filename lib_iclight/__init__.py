VERSION = "2.0"

t2i_fc: str = """
<ins><b>Relighting with Foreground Condition</b></ins><br>
Given a foreground image, generate a new background via <code>txt2img</code>,
then blend them together with coherent lighting conditions.
"""

t2i_fbc: str = """
<ins><b>Relighting with Foreground and Background Condition</b></ins><br>
Extract the subject from the foreground image, then blend it onto the background image,
while keeping the lighting conditions coherent.<br>
<code>Sampler</code> and <code>Steps</code> are important; <code>Prompts</code> doesn't matter.
"""

i2i_fc: str = """
<ins><b>Relighting with Light-Map Condition</b></ins><br>
Given an input image, generate a new background using conditioned lighting.
"""

removal: str = """
<i><b>Note:</b> Disable this feature if the image already has no background</i>
"""

raw: str = """
Use the input before the background removal as the "Original" to restore details from
"""
