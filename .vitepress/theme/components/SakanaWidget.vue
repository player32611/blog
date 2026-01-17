<template>
    <div id="sakana-widget"></div>
</template>

<script>
import 'sakana-widget/lib/index.css';

export default {
    name: "SakanaWidget",
    async mounted() {
        const [{ default: SakanaWidget }, KoishiImage1] = await Promise.all([
            import('sakana-widget'),
            import('/images/Components/koishi.gif?url')  // Vite 会把 ?url 转成字符串
        ]);

        const Koishi1 = SakanaWidget.getCharacter('takina');
        Koishi1.image = KoishiImage1.default || KoishiImage1; // 兼容两种返回
        SakanaWidget.registerCharacter('Koishi1', Koishi1);

        new SakanaWidget({
            size: 200,
            character: 'Koishi1'
        }).mount('#sakana-widget');
    }
}
</script>

<style scoped></style>