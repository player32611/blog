<template>
    <div id="sakana-widget"></div>
</template>

<script>
    import 'sakana-widget/lib/index.css';
    import SakanaWidget from 'sakana-widget';
    export default {
        name:"SakanaWidget",
        mounted(){
            new SakanaWidget().mount('#sakana-widget');
        }
    }
</script>

<style scoped>

</style>