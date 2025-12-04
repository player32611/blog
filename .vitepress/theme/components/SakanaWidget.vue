<template>
    <div id="sakana-widget"></div>
</template>

<script>
    import 'sakana-widget/lib/index.css';
    import SakanaWidget from 'sakana-widget';

    import KoishiImage1 from '/images/Components/Koishi-1.gif';
    const Koishi1 = SakanaWidget.getCharacter('takina');
    Koishi1.image = KoishiImage1;
    SakanaWidget.registerCharacter('Koishi1', Koishi1);
    
    export default {
        name:"SakanaWidget",
        mounted(){
            new SakanaWidget({ 
                size: 200,
                character: 'Koishi1'
            }).mount('#sakana-widget');
        }
    }
</script>

<style scoped>

</style>